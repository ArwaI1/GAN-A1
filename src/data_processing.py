import os
import zipfile
import tensorflow as tf

def download_and_extract_data(raw_dir='data/raw', processed_dir='data/processed'):
    """Downloads Medical MNIST from Kaggle and extracts it."""
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    zip_path = os.path.join(raw_dir, 'medical-mnist.zip')
    
    if not os.path.exists(zip_path):
        print("Downloading dataset from Kaggle...")
        os.system(f'kaggle datasets download -d andrewmvd/medical-mnist -p "{raw_dir}"')
    
    if len(os.listdir(processed_dir)) == 0:
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(processed_dir)
        print("Dataset extracted!")
    else:
        print("Dataset already extracted locally.")
        
    # Find the exact folder containing the images
    actual_data_dir = None
    for root, dirs, files in os.walk(processed_dir):
        if any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in files):
            actual_data_dir = os.path.dirname(root)
            break
            
    return actual_data_dir

def normalize(image):
    return tf.cast(image, tf.float32) / 255.0

def add_noise(image):
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.15)
    return tf.clip_by_value(image + noise, 0.0, 1.0), image

def create_datasets(data_dir, batch_size=64, img_size=(64, 64)):
    """Creates standard and noisy tf.data pipelines."""
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir, labels=None, color_mode='grayscale',
        image_size=img_size, batch_size=batch_size, shuffle=True
    )
    
    normalized_ds = dataset.map(normalize)
    train_ds = normalized_ds.map(lambda x: (x, x)).prefetch(tf.data.AUTOTUNE)
    noisy_train_ds = normalized_ds.map(add_noise).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, noisy_train_dsNew-Item -ItemType File -Force -Path .gitignore