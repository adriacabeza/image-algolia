import argparse
import os
import pickle

import h5py
import numpy as np
import numpy.typing as npt

import src.util.log_util as log
from src.extract_features.extract import encode_search_query
from src.util.time_util import timing


@timing
def find_best_matches(text_features: npt.ArrayLike, image_features: npt.ArrayLike):
    # Compute the similarity between the search query and each photo using the Cosine similarity
    similarities = (image_features @ text_features.T).squeeze(1)

    # Sort the photos by their similarity score
    # TODO use a threshold to approach empty results
    best_photo_idx = (-similarities).argsort()
    return best_photo_idx


@timing
def search(search_query: str, photo_features: npt.ArrayLike, photo_ids: list, results_count: int) -> np.ndarray:
    # Encode the search query
    text_features = encode_search_query(search_query)

    # Find the best matches
    best_photos = find_best_matches(text_features, photo_features)
    result = [photo_ids[i] for i in best_photos[:results_count]]
    log.info(f'Result:{result}')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', type=str, required=True)
    parser.add_argument('--results_count', type=int, default=1)
    parser.add_argument('--features', type=str, required=True)
    parser.add_argument('--photo_ids', type=str, required=True)
    args = parser.parse_args()

    assert os.path.exists(args.features)
    assert os.path.exists(args.photo_ids)
    with h5py.File(args.features, 'r') as hf:
        photo_features = hf['features'][:]

    log.info(f'Searching for {args.query} in our image database')
    with open(args.photo_ids) as f:
        photo_ids = f.readlines()
    search(args.query, photo_features, photo_ids, args.results_count)
