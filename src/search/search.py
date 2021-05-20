import argparse
import os
import pickle

import h5py
import numpy as np
import numpy.typing as npt

import src.util.log_util as log
from src.extract_features.extract import encode_search_query, extract_new_images, encode_all_images, get_images
from src.util.time_util import timing


@timing
def find_best_matches(text_features: npt.ArrayLike, image_features: npt.ArrayLike):
    # Compute the similarity between the search query and each photo using the Cosine similarity
    similarities = (image_features @ text_features.T).squeeze(1)
    # Sort the photos by their similarity score
    # TODO use a threshold to approach empty results
    best_photo_idx = (-similarities).argsort()
    return best_photo_idx, similarities


@timing
def search(search_query: str, photo_features: npt.ArrayLike, photo_ids: list, results_count: int) -> np.ndarray:
    # Encode the search query
    text_features = encode_search_query(search_query)
    # Find the best matches
    best_photos, similarities = find_best_matches(text_features, photo_features)
    print(similarities)
    result = [(photo_ids[i], similarities[i]) for i in best_photos[:results_count]]
    log.info(f'Result:{result}')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', type=str, required=True)
    parser.add_argument('--results_count', type=int, default=1)
    parser.add_argument('--features', type=str, required=True)
    parser.add_argument('--photo_ids', type=str, required=True)
    parser.add_argument('--images', type=str)
    args = parser.parse_args()

    if not os.path.exists(args.features) or not os.path.exists(args.photo_ids):
        if args.images:
            log.info('Features not found. Extracting first')
            list_of_images = get_images(args.images)
            encode_all_images(list_of_images, 8, args.photo_ids)
        else:
            raise Exception("You need to specify the photos path if the features do not exist")

    if args.images:
        log.info('Extracting new images inserted to the database')
        extract_new_images(args.images, args.photo_ids)

    with open(args.photo_ids) as f:
        photo_ids = f.readlines()

    with h5py.File(args.features, 'r') as hf:
        photo_features = hf['features'][:]

    log.info(f'Searching for {args.query} in our image database')
    search(args.query, photo_features, photo_ids, args.results_count)
