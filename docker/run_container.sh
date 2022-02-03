SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

function usage {
  echo "USAGE: $0 [--quantum] [--no-gpu]"
  exit
}

cd $SCRIPTPATH

QUANTUM=0
GPU=1

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

if (($QUANTUM)); then
  if (($GPU)); then
    docker run --gpus all --rm -u $(id -u):$(id -g) -v $(pwd)/..:/home/tami -it tami_exp_quantum/tensorflow:latest bash
  else
    docker run --rm -u $(id -u):$(id -g) -v $(pwd)/..:/home/tami -it tami_exp_quantum/tensorflow:latest bash
  fi
else
  if (($GPU)); then
    docker run --gpus all --rm -u $(id -u):$(id -g) -v $(pwd)/..:/home/tami -it tami_exp/tensorflow:latest bash
  else
    docker run --rm -u $(id -u):$(id -g) -v $(pwd)/..:/home/tami -it tami_exp/tensorflow:latest bash
  fi
fi