SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd ${SCRIPTPATH}

#docker save -o <path for generated tar file> <image name>
wget "https://ocsdev-cloud.duckdns.org/index.php/s/JJpd7CiNJryYQzG/download/tami_expV1.0"
docker load -i "tami_expV1.0"

echo "Load completed, you can now run run_container.sh"