SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

cd $SCRIPTPATH

sudo docker run --gpus all -u $(id -u):$(id -g) -v $(pwd)/..:/home/tami -it tami_exp/tensorflow:latest bash
