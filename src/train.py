import os
import tensorflow as tf
import matplotlib.pyplot as plt
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

def main():
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # 1. Data Pipeline
    print("--- 1. SETTING UP DATA ---")
    data_dir = download_and_extract_data()
    train_ds, noisy_train_ds = create_datasets(data_dir)
    sample_batch = next(iter(train_ds))[0][:5]
    
    # 2. Train Standard AE
    print("--- 2. TRAINING STANDARD AE ---")
    ae, ae_encoder, ae_decoder = build_autoencoder(latent_dim=64)
    ae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    ae.fit(train_ds, epochs=6) 
    
    ae.save_weights('models/ae_weights.weights.h5')
    
    ae_reconstructions = ae.predict(sample_batch)
    plot_and_save(sample_batch, ae_reconstructions, "Standard Autoencoder Results", "ae_results.png")

    # 3. Train Denoising AE
    print("--- 3. TRAINING DENOISING AE ---")
    ae.fit(noisy_train_ds, epochs=6)
    
    ae.save_weights('models/dae_weights.weights.h5')
    
    noisy_batch, clean_batch = next(iter(noisy_train_ds))
    denoised_batch = ae.predict(noisy_batch[:5])
    plot_and_save(noisy_batch[:5], denoised_batch, "Denoising Autoencoder Results", "dae_results.png", ("Noisy Input", "Denoised"))

    # 4. Train VAE
    print("--- 4. TRAINING VAE ---")
    vae_encoder, vae_decoder = build_vae_components(latent_dim=64)
    vae = VAE(vae_encoder, vae_decoder)
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005))
    vae.fit(train_ds, epochs=10)
    
    # Pass a batch of data through the model to "build" it before saving ---
    vae(sample_batch) 
    
    vae.save_weights('models/vae_weights.weights.h5')
    
    _, _, z_sample = vae.encoder.predict(sample_batch)
    vae_reconstructions = vae.decoder.predict(z_sample)
    plot_and_save(sample_batch, vae_reconstructions, "VAE Results", "vae_results.png")

    # 5. Generative Samples & Latent Space
    print("--- 5. GENERATING SAMPLES & PLOTTING LATENT SPACE ---")
    
    # Sample Generation
    random_latent_vectors = tf.random.normal(shape=(5, 64))
    generated_images = vae.decoder.predict(random_latent_vectors)
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        axes[i].imshow(generated_images[i].squeeze(), cmap='gray')
        axes[i].axis('off')
    plt.suptitle("New Samples Generated from VAE Latent Space")
    plt.savefig('results/vae_generated_samples.png')
    plt.close()

    # Latent Space Visualization
    plot_batches = [next(iter(train_ds))[0] for _ in range(10)]
    plot_data = tf.concat(plot_batches, axis=0)
    ae_embeddings = ae_encoder.predict(plot_data)
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(ae_embeddings)

    plt.figure(figsize=(8, 6))
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.6, s=15)
    plt.title("2D Visualization of AE Latent Space (PCA)")
    plt.savefig('results/latent_space_pca.png')
    plt.close()

    print("Pipeline complete! All weights saved to 'models/' and plots saved to 'results/'.")

if __name__ == "__main__":
    main()