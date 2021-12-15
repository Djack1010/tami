SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

cd $SCRIPTPATH
cp ../requirements.txt .

docker build -t tami_exp/tensorflow:latest .

rm -f requirements.txt