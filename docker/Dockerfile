FROM tensorflow/tensorflow:latest-gpu
MAINTAINER Giacomo Iadarola "giacomo.iadarola@iit.cnr.it"
WORKDIR /home/tami
ADD requirements.txt /home/tami/requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install -r requirements.txt
