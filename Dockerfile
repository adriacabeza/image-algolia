FROM python:latest

WORKDIR /srv/image_algolia

RUN mkdir features

COPY requirements.lock  .
RUN pip3 install -r requirements.lock --no-cache-dir

ADD src ./src

CMD  python3 -m src.search.search --query {QUERY}  --features features/features.h5 --photo_ids photo_ids.csv  --images data
