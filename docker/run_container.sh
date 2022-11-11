SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

function usage {
  echo "USAGE: $0 [--quantum] [--no-gpu]"
  if [ "$1" == "--no-exit" ]; then
    :
  else
    exit
  fi
}

cd $SCRIPTPATH

usage --no-exit

if nvidia-smi &>/dev/null ; then
  GPU=1
else
  echo "Nvidia drivers not found, running in CPU-mode. For more information, see https://github.com/NVIDIA/nvidia-docker"
  GPU=0
fi

for arg in "$@"; do
  if [ "$arg" == "--quantum" ]; then
    QUANTUM=1
  elif [ "$arg" == "--no-gpu" ]; then
    GPU=0
  else
    echo "ERROR! Incorrect parameter '$arg'"
    usage 
  fi 
done

echo "STARTING Tensorflow container"

if (($QUANTUM)); then
  if (($GPU)); then
    echo "QUANTUM: true; GPU: true;"
    docker run --gpus all --rm -u $(id -u):$(id -g) -v $(pwd)/..:/home/tami --name="${USER}_$(date +%s)" -it tami_exp_quantum/tensorflow:2.7.0 bash
  else
    echo "QUANTUM: true; GPU: false;"
    docker run --rm -u $(id -u):$(id -g) -v $(pwd)/..:/home/tami --name="${USER}_$(date +%s)" -it tami_exp_quantum/tensorflow:2.7.0 bash
  fi
else
  if (($GPU)); then
    echo "QUANTUM: false; GPU: true;"
    docker run --gpus all --rm -u $(id -u):$(id -g) -v $(pwd)/..:/home/tami --name="${USER}_$(date +%s)" -it tami_exp/tensorflow:2.7.0 bash
  else
    echo "QUANTUM: false; GPU: false;"
    docker run --rm -u $(id -u):$(id -g) -v $(pwd)/..:/home/tami --name="${USER}_$(date +%s)" -it tami_exp/tensorflow:2.7.0 bash
  fi
fi