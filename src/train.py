import os
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from data_processing import download_and_extract_data, create_datasets
from model import build_autoencoder, build_vae_components, VAE

def plot_and_save(images1, images2, title, filename, titles=("Original", "Reconstruction")):
    """Helper to save comparison plots."""
    fig, axes = plt.subplots(2, 5, figsize=(15, 4))
    for i in range(5):
        axes[0, i].imshow(images1[i].numpy().squeeze(), cmap='gray')
        axes[0, i].set_title(titles[0])
        axes[0, i].axis('off')
        
        # Handle cases where output is eager tensor or numpy array
        img2 = images2[i].numpy() if hasattr(images2[i], 'numpy') else images2[i]
        axes[1, i].imshow(img2.squeeze(), cmap='gray')
        axes[1, i].set_title(titles[1])
        axes[1, i].axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f'results/{filename}')
    plt.close()

def train_models_for_region(region_name, region_dir):
    print(f"==================================================")
    print(f"=== TRAINING MODELS FOR REGION: {region_name} ===")
    print(f"==================================================")

    # 1. Data Pipeline
    print(f"--- 1. SETTING UP DATA FOR {region_name} ---")
    train_ds, noisy_train_ds = create_datasets(region_dir)
    sample_batch = next(iter(train_ds))[0][:5]
    
    # 2. Train Standard AE
    print(f"--- 2. TRAINING STANDARD AE FOR {region_name} ---")
    ae, ae_encoder, ae_decoder = build_autoencoder(latent_dim=64)
    ae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    hist_ae = ae.fit(train_ds, epochs=3, verbose=2) 
    
    ae.save_weights(f'models/{region_name}_ae_weights.weights.h5')
    
    ae_reconstructions = ae.predict(sample_batch)
    plot_and_save(sample_batch, ae_reconstructions, f"Standard AE Results - {region_name}", f"{region_name}_ae_results.png")

    # 3. Train Denoising AE
    print(f"--- 3. TRAINING DENOISING AE FOR {region_name} ---")
    hist_dae = ae.fit(noisy_train_ds, epochs=3, verbose=2)
    
    ae.save_weights(f'models/{region_name}_dae_weights.weights.h5')
    
    noisy_batch, clean_batch = next(iter(noisy_train_ds))
    denoised_batch = ae.predict(noisy_batch[:5])
    plot_and_save(noisy_batch[:5], denoised_batch, f"Denoising AE Results - {region_name}", f"{region_name}_dae_results.png", ("Noisy Input", "Denoised"))

    # 4. Train VAE
    print(f"--- 4. TRAINING VAE FOR {region_name} ---")
    vae_encoder, vae_decoder = build_vae_components(latent_dim=64)
    vae = VAE(vae_encoder, vae_decoder)
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005))
    hist_vae = vae.fit(train_ds, epochs=5, verbose=2)
    
    # Pass a batch of data through the model to "build" it before saving ---
    vae(sample_batch) 
    
    vae.save_weights(f'models/{region_name}_vae_weights.weights.h5')
    
    _, _, z_sample = vae.encoder.predict(sample_batch)
    vae_reconstructions = vae.decoder.predict(z_sample)
    plot_and_save(sample_batch, vae_reconstructions, f"VAE Results - {region_name}", f"{region_name}_vae_results.png")

    # 5. Generative Samples & Latent Space
    print(f"--- 5. GENERATING SAMPLES & PLOTTING LATENT SPACE FOR {region_name} ---")
    
    # Sample Generation
    random_latent_vectors = tf.random.normal(shape=(5, 64))
    generated_images = vae.decoder.predict(random_latent_vectors)
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        axes[i].imshow(generated_images[i].squeeze(), cmap='gray')
        axes[i].axis('off')
    plt.suptitle(f"New Samples from VAE Latent Space - {region_name}")
    plt.savefig(f'results/{region_name}_vae_generated_samples.png')
    plt.close()

    # Latent Space Visualization
    plot_batches = [next(iter(train_ds))[0] for _ in range(10)]
    plot_data = tf.concat(plot_batches, axis=0)
    ae_embeddings = ae_encoder.predict(plot_data)
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(ae_embeddings)

    plt.figure(figsize=(8, 6))
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.6, s=15)
    plt.title(f"2D Visualization of AE Latent Space (PCA) - {region_name}")
    plt.savefig(f'results/{region_name}_latent_space_pca.png')
    plt.close()

    return {
        'region': region_name,
        'ae_loss': hist_ae.history['loss'][-1],
        'dae_loss': hist_dae.history['loss'][-1],
        'vae_total_loss': hist_vae.history['loss'][-1],
        'vae_reconstruction_loss': hist_vae.history['reconstruction_loss'][-1],
        'vae_kl_loss': hist_vae.history['kl_loss'][-1]
    }

def main():
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # 1. Data Pipeline
    print("--- 1. SETTING UP DATA ---")
    # Returns the base directory (e.g. data/processed) where region subdirectories are located
    data_dir = download_and_extract_data()
    
    # Identify anatomical regions (subdirectories in data_dir)
    regions = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print(f"Found regions: {regions}")
    
    all_results = []
    
    for region in regions:
        region_dir = os.path.join(data_dir, region)
        region_results = train_models_for_region(region, region_dir)
        all_results.append(region_results)
        
        # Clear keras session to avoid memory leak and slow down
        tf.keras.backend.clear_session()
        
    df = pd.DataFrame(all_results)
    df.to_csv('results/training_metrics.csv', index=False)

    print("Pipeline complete! All weights, plots, and metrics (in training_metrics.csv) saved.")

if __name__ == "__main__":
    main()