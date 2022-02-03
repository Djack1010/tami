SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

function usage {
    echo "USAGE: $0 [--quantum]"
    exit
}

cd $SCRIPTPATH

QUANTUM=0

for arg in "$@"; do
    if [ "$arg" == "--quantum" ]; then
        QUANTUM=1
    else
        echo "ERROR! Incorrect parameter '$arg'"
        usage 
    fi 
done

if (($QUANTUM)); then
    cp ../full_requirements.txt ./requirements.txt
    docker build -t tami_exp_quantum/tensorflow:latest .
else
    cp ../partial_requirements.txt ./requirements.txt
    docker build -t tami_exp/tensorflow:latest .
fi

rm -f requirements.txt
