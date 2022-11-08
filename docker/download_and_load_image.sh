SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd ${SCRIPTPATH}

URLTAMI="https://martellone.iit.cnr.it/index.php/s/8qA8jtiK5SYtMk9/download/tami_expV1.0.tar"

#docker save -o <path for generated tar file> <image name>
echo "Downloading TAMI image from ocsdev cloud"
wget ${URLTAMI}
echo "Loading TAMI image in local docker instance (it may take a while...)"
NAMETAMI=$(basename ${URLTAMI})
docker load -i ${NAMETAMI}

echo "Load completed, you can now run the script 'run_container.sh'"