#!/bin/bash

srcdir=`dirname $0`
. "${srcdir}/config"
. "${srcdir}/lib.sh"
check_config "IMAGE COMMON_IMAGE"

set -e
set -x

# login to aws
aws ecr get-login --no-include-email | bash

#docker build -t ${COMMON_IMAGE} \
#  -f ${srcdir}/Dockerfile.base ${srcdir}/..
#docker push ${COMMON_IMAGE}
docker pull ${COMMON_IMAGE}

docker build -t ${IMAGE} -f ${srcdir}/Dockerfile --build-arg COMMON_IMAGE=${COMMON_IMAGE} ${srcdir}/..
docker push ${IMAGE}
