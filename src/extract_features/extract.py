import argparse
import glob
import os
import pandas as pd

import clip
import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import src.util.log_util as log
from src.util.time_util import timing

# Load the open CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


@timing
def encode_search_query(search_query: list):
    with torch.no_grad():
        # Encode and normalize the search query using CLIP
        text_encoded = model.encode_text(clip.tokenize(search_query).to(device))
        text_encoded /= text_encoded.norm(dim=-1, keepdim=True)
    # Retrieve the feature vector
    return text_encoded.cpu().numpy()


def encode_image_batch(photos_batch: list):
    # Load all the photos from the files
    photos = [Image.open(photo_file) for photo_file in photos_batch]
    # Preprocess all photos
    photos_preprocessed = torch.stack([preprocess(photo) for photo in photos]).to(device)

    with torch.no_grad():
        # Encode the photos batch to compute the feature vectors and normalize them
        photos_features = model.encode_image(photos_preprocessed)
        photos_features /= photos_features.norm(dim=-1, keepdim=True)

    # Transfer the feature vectors back to the CPU and convert to numpy
    return photos_features.cpu().numpy()


@timing
def extract_new_images(images_path, photo_ids):
    with open(photo_ids) as f:
        photos = f.readlines()
    list_of_images = get_images(images_path)
    new_list = list(set(list_of_images) - set(photos))
    log.info(f'New {len(new_list)} images found. Starting to extract')
    encode_all_images(new_list, 8, photo_ids)


def get_images(images_path: str):
    list_of_images = glob.glob(f'{images_path}/*.jpg') +\
                     glob.glob(f'{images_path}/*.jpeg') + \
                     glob.glob(f'{images_path}/*.png') + glob.glob(f'{images_path}/*.JPG')

    return list_of_images


@timing
def encode_all_images(list_of_images: list, batch_size: int, photo_ids: str):
    length = len(list_of_images)
    assert length > 0
    with tqdm(total=length) as progress_bar:
        for i in range(0, len(list_of_images), batch_size):
            batch = list_of_images[i:i+batch_size]
            batch_features = encode_image_batch(batch)
            progress_bar.update(batch_size)
            np.save(f'features/batch_{i}_{i + batch_size}.npy', batch_features)
            photos = [filename for filename in batch]
            photo_ids_data = pd.DataFrame(photos, columns=['photo_id'])
            photo_ids_data.to_csv(f'features/batch_{i}_{i + batch_size}.csv', index=False)

    log.info('Features extracted')
    csv_files = sorted(glob.glob("features/*.csv"))
    photos_df = pd.concat([pd.read_csv(ids_file) for ids_file in csv_files])
    if os.path.exists(photo_ids):
        log.info('Merging previous photo_ids with the latest update')
        previous_photo_ids = pd.read_csv(photo_ids)
        photo_df = pd.concat([previous_photo_ids, photos_df])
    photos_df.to_csv(photo_ids, index=False)

    log.info('Merging batch features')
    batch_files = sorted(glob.glob('features/*.npy'))
    features = np.concatenate([np.load(features_file) for features_file in batch_files])
    log.info(f'Features computed of shape {features.shape}')
    if os.path.exists('features/features.h5'):
        log.info('Merging previous features with the latest update')
        with h5py.File('features/features.h5', 'r') as hf:
            previous_photo_features = hf['features'][:]
        features = np.concatenate([previous_photo_features, features])

    with h5py.File(f'features/features.h5', 'w') as hf:
        hf.create_dataset('features', data=features)

    # removing useless batch feature file
    for batch_file in batch_files:
        os.remove(batch_file)
    for csv_file in csv_files:
        os.remove(csv_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--images', type=str, required=True)
    parser.add_argument('--photo_ids', type=str, default='photo_ids.csv')
    args = parser.parse_args()

    assert os.path.exists(args.images)
    assert args.batch_size > 0

    os.makedirs('features', exist_ok=True)
    log.info(f'Starting to process all the images from {args.images}')
    encode_all_images(args.images, args.batch_size, args.photo_ids)
