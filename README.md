<h1 align="center"> Algolia for images: search images by content </h1>
[![Python application](https://github.com/adriacabeza/image-algolia/actions/workflows/python-app.yml/badge.svg)](https://github.com/adriacabeza/image-algolia/actions/workflows/python-app.yml)

Using the [CLIP model](https://arxiv.org/pdf/2103.00020.pdf) from OpenAI to build a service to search for images in a human way. 

## Setup
Clone the repository, create an environment and install all the dependencies.
> this was tested using Python 3.9.4

```bash
# Create environment
python3 -m venv env
source env/bin/activate
# Install dependencies
pip install git+https://github.com/openai/CLIP.git
pip install -r requirements.lock
```

### 1 - Collect some images

They can be from your personal library or from a public dataset. In my case I have used the [Unsplash Lite Dataset](https://github.com/unsplash/datasets) containing 25k photos. You can use the script **download_images.py** available in the dataset folder which downloads in a thread pool all the images.


### 2 - Extract features from an image folder

If it is your first time running it, first it will download the weights of the model which can take from 1-2 minutes. 
```bash
python3 -m src.extract_features.extract --images data
```

### 3 - Search for a text query

The previous method will create a hdf5 file in the features folder containing the images encoded, and the photo_ids.txt file which contains the list of images in the same order as in the encoded array. This is handy when it comes to return an image result. 

```bash
python3 -m src.search.search --query car --features features/features.h5 --photo_ids photo_ids.txt  
```

