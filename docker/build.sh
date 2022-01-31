SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

cd $SCRIPTPATH

if [ "$1" == "--quantum" ]; then
  cp ../quantum_requirements.txt ./requirements.txt
else
  cp ../requirements.txt .
fi

docker build -t tami_exp/tensorflow:latest .

rm -f requirements.txt