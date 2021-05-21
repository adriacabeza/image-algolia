import argparse
import os

import h5py
import numpy as np
import numpy.typing as npt

import src.util.log_util as log
from src.extract_features.extract import encode_search_query, extract_new_images, encode_all_images, get_images
from src.util.time_util import timing


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def compute_similarity(text_features: npt.ArrayLike, image_features: npt.ArrayLike):
    # Compute the similarity between the search query and each photo using the Cosine similarity
    similarities = 100*(text_features @ image_features.T)
    return softmax(similarities[0])


@timing
def search(search_query: str, photo_features: npt.ArrayLike, photo_ids: list, results_count: int) -> np.ndarray:
    # Encode the search query
    text_features = encode_search_query(search_query)
    # Find the best matches
    similarities = compute_similarity(text_features, photo_features)
    best_match = sorted(zip(similarities, range(len(photo_features))), key=lambda x: x[0], reverse=True) # TODO argsort
    for i in range(results_count):
        log.info(f'{i}: {photo_ids[best_match[i][1]].rstrip()} with a similarity of {round(100*best_match[i][0],3)}%')


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
            log.info('Features or photo ids not found. Extracting first')
            list_of_images = get_images(args.images)
            encode_all_images(list_of_images, 8, args.photo_ids)
        else:
            raise Exception("You need to specify the photos path if the features do not exist")
    elif args.images:
        log.info('Checking for new images inserted to the database')
        extract_new_images(args.images, args.photo_ids)

    with open(args.photo_ids) as f:
        photo_ids = f.readlines()

    with h5py.File(args.features, 'r') as hf:
        photo_features = hf['features'][:]

    assert args.results_count <= len(photo_ids)
    log.info(f'Searching for {args.query} in our image database of {len(photo_ids)} images')
    search(args.query, photo_features, photo_ids, args.results_count)
