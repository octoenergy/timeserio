TF_VERSION=2.8.0
PY_VERSION?=py39
CPU_IMAGE?=tensorflow:${TF_VERSION}-${PY_VERSION}
GPU_IMAGE?=tensorflow:${TF_VERSION}-gpu-${PY_VERSION}
DOCKER_NAMESPACE?=krakentechnologies

.PHONY: all build-cpu build-gpu push-cpu push-gpu

all: build-cpu build-gpu push-cpu push-gpu

build-cpu: 
	docker build -t ${CPU_IMAGE} -f ./Dockerfile.cpu . --build-arg TF_PACKAGE="tensorflow==${TF_VERSION}"

build-gpu:
	docker build -t ${GPU_IMAGE} -f ./Dockerfile.gpu . --build-arg TF_PACKAGE="tensorflow-gpu==${TF_VERSION}"

push-cpu:
	docker login
	docker tag ${CPU_IMAGE} ${DOCKER_NAMESPACE}/${CPU_IMAGE}
	docker push ${DOCKER_NAMESPACE}/${CPU_IMAGE}

push-gpu:
	docker login
	docker tag ${GPU_IMAGE} ${DOCKER_NAMESPACE}/${GPU_IMAGE}
	docker push ${DOCKER_NAMESPACE}/${GPU_IMAGE}
