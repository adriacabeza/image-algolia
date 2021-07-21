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
> python3 -m src.extract_features.extract --images data
{image_algolia} - <2021-05-21 12:04:10,227> - [INFO   ] - Starting to process all the images from data
960it [01:44,  9.18it/s]
{image_algolia} - <2021-05-21 12:05:54,821> - [INFO   ] - Features extracted
{image_algolia} - <2021-05-21 12:05:55,037> - [INFO   ] - Merging batch features
{image_algolia} - <2021-05-21 12:05:55,144> - [INFO   ] - Features computed of shape (959, 512)
{image_algolia} - <2021-05-21 12:05:55,144> - [INFO   ] - Merging previous features with the latest update
{image_algolia} - <2021-05-21 12:05:55,200> - [INFO   ] - func:'encode_all_images' took: 104.9571 sec
```

### 3 - Search for a text query

The previous method will create a hdf5 file in the features folder containing the images encoded, and the photo_ids.txt file which contains the list of images in the same order as in the encoded array. This is handy when it comes to return an image result. **In the case where you call this method without any feature extracted previously it will automatically extract them. Also if you add new images it will detect them and only extract those before the search**.  

```bash
> python3 -m src.search.search --query "a forrest"  --features features/features.h5 --photo_ids photo_ids.csv  --images data
{image_algolia} - <2021-05-21 11:49:00,652> - [INFO   ] - Features or photo ids not found. Extracting first
960it [02:08,  7.49it/s]
{image_algolia} - <2021-05-21 11:51:08,860> - [INFO   ] - Features extracted
{image_algolia} - <2021-05-21 11:51:09,129> - [INFO   ] - Merging batch features
{image_algolia} - <2021-05-21 11:51:09,252> - [INFO   ] - Features computed of shape (959, 512)
{image_algolia} - <2021-05-21 11:51:09,289> - [INFO   ] - func:'encode_all_images' took: 128.6155 sec
{image_algolia} - <2021-05-21 11:51:09,295> - [INFO   ] - Searching for a forrest in our image database of 959 images
{image_algolia} - <2021-05-21 11:51:09,671> - [INFO   ] - func:'encode_search_query' took: 0.3757 sec
{image_algolia} - <2021-05-21 11:51:09,674> - [INFO   ] - 0: data/4mNV6RJcEu8.jpg with a similarity of 7.879%
{image_algolia} - <2021-05-21 11:51:09,674> - [INFO   ] - func:'search' took: 0.3792 sec
```


## Containers solve all problems
 
You can run it using docker-compose. It will automatically extract first the features and then search for the QUERY present in the docker-compose YAML. 

```
docker-compose -f docker-compose.test.yml up --build
```
