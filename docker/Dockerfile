FROM tensorflow/tensorflow:2.7.0-gpu
MAINTAINER Giacomo Iadarola "giacomo.iadarola@iit.cnr.it"
WORKDIR /home/tami
#IF there are some ubuntu repo not updated, uncomment the next 2 lines to fix them
#RUN echo "deb http://archive.ubuntu.com/ubuntu/ $(lsb_release -sc) main universe Trusty-updates" >> /etc/apt/sources.list
#RUN echo exit 101 > /usr/sbin/policy-rc.d && chmod +x /usr/sbin/policy-rc.d
RUN apt-get update && apt-get -y --fix-missing install ffmpeg libsm6 libxext6
RUN python -m pip install --upgrade pip setuptools wheel
ADD requirements.txt /home/tami/requirements.txt
RUN python -m pip install -r quantum_requirements.txt