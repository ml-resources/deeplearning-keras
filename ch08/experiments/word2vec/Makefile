.PHONY build compile run-it test train predict clean

TRAIN_SET_URL="http://mattmahoney.net/dc/text8.zip"

DIR=$(shell pwd)
DATA_DIR=$(DIR)/data
DOCKER_IMAGE="manpreet/keras"
CONTAINER_NAME="word2vec-keras"
DOCKER_RUN=docker run --rm --name $(CONTAINER_NAME) -v $(DIR):/srv/ai -w /srv/ai/src

build:
docker build . -t $(DOCKER_IMAGE)

compile:
$(DOCKER_RUN) $(DOCKER_IMAGE) python3 setup.py build_ext --inplace

run-it:
$(DOCKER_RUN) -it $(DOCKER_IMAGE) /bin/bash

test:
$(DOCKER_RUN) $(DOCKER_IMAGE) python3 -m unittest

train:
