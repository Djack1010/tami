FROM tensorflow/tensorflow:2.7.0-gpu
MAINTAINER Giacomo Iadarola "giacomo.iadarola@iit.cnr.it"
WORKDIR /home/tami
#IF there are some ubuntu repo not updated, uncomment the next 2 lines to fix them
RUN echo "deb http://archive.ubuntu.com/ubuntu/ $(lsb_release -sc) main universe Trusty-updates" >> /etc/apt/sources.list
RUN echo exit 101 > /usr/sbin/policy-rc.d && chmod +x /usr/sbin/policy-rc.d
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt install wget -y
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
RUN apt-key del 7fa2af80
RUN dpkg -i cuda-keyring_1.0-1_all.deb
RUN apt-get update && apt-get upgrade -y
RUN apt-get update && apt-get -y --fix-missing install ffmpeg libsm6 libxext6
RUN apt-get -y install sox libcairo2 libcairo2-dev
RUN apt-get -y --fix-missing install openjdk-11-jre-headless
RUN python -m pip install --upgrade pip setuptools wheel
ADD requirements.txt /home/tami/requirements.txt
RUN python -m pip install -r /home/tami/requirements.txt
# Install Java 11 (already installed) and APKTool -> only for running CATI on container
#  also commented lines in ./manual_build.sh has to be restored
#RUN apt-get -y install openjdk-11-jre-headless
#ADD apktool /home/tami/apktool
#RUN chmod +x apktool/apktool*
#RUN mv apktool/apktool* /usr/local/bin/
