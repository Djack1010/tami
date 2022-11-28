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
    cp ../requirements/full_requirements.txt ./requirements.txt
    docker build -t tami_exp_quantum/tensorflow:2.7.0 .
else
    cp ../requirements/partial_requirements.txt ./requirements.txt
    #Uncomment the next line for installing also APKTOOL and being able to run CATI
    # Also, the Dockerfile has to be changed (and apktool files has to be placed in ext_tools/apktool)
    #cp -r ../ext_tools/apktool .
    docker build -t tami_exp/tensorflow:2.7.0 .
fi

#rm -rf apktool
rm -f requirements.txt
