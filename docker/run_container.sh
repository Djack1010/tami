SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

cd $SCRIPTPATH

if [ "$1" == "--no-gpu" ]; then
  docker run --rm -u $(id -u):$(id -g) -v $(pwd)/..:/home/tami -it tami_exp/tensorflow:latest bash
else
  docker run --gpus all --rm -u $(id -u):$(id -g) -v $(pwd)/..:/home/tami -it tami_exp/tensorflow:latest bash
fi
