import argparse
import os
import urllib.request
from multiprocessing.pool import ThreadPool

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--images_csv', type=str, required=True)
parser.add_argument('--data_path', type=str, required=True)
args = parser.parse_args()

assert os.path.exists(args.images_csv)
os.makedirs(args.data_path, exist_ok=True)


def download_photo(photo: str):
    photo_id = photo[0]
    photo_url = photo[1] + "?w=640"

    # Path where the photo will be stored
    photo_path = os.path.join(args.data_path, f"{photo_id}.jpg")

    # Only download a photo if it doesn't exist
    if not os.path.exists(photo_path):
        try:
            urllib.request.urlretrieve(photo_url, photo_path)
        except:
            # Catch the exception if the download fails for some reason
            print(f"Cannot download {photo_url}")
            pass


photos = pd.read_csv(args.images_csv, sep='\t', header=0)
photo_urls = photos[['photo_id', 'photo_image_url']].values.tolist()
print(f'Photos in the dataset: {len(photo_urls)}')
threads_count = 16
pool = ThreadPool(threads_count)
pool.map(download_photo, photo_urls)
print(f'Photos downloaded: {len(photo_urls)}')
