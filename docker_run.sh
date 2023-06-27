# GPU support: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
# if not existing, create folders
mkdir -p results
mkdir -p datasets
    
#--rm    \
# docker run \

docker run \
    -d \
    --memory=32g \
    --gpus all \
    --user $UID:$GID \
    --mount src="$(pwd)/datasets",target=/datasets,type=bind \
    --mount src="$(pwd)/results",target=/results,type=bind \
    --mount src="$(pwd)/run.sh",target=/run.sh,type=bind \
    sati