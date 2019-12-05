#!/bin/bash

srcdir=`dirname $0`
. "${srcdir}/config"
. "${srcdir}/lib.sh"
check_config "IMAGE COMMON_IMAGE genie_version"

set -e
set -x

#docker build -t ${COMMON_IMAGE} \
#  -f ${srcdir}/Dockerfile.base ${srcdir}/..
#docker push ${COMMON_IMAGE}
docker pull ${COMMON_IMAGE}

docker build -t ${IMAGE} -f ${srcdir}/Dockerfile \
  --build-arg COMMON_IMAGE=${COMMON_IMAGE} \
  --build-arg GENIE_VERSION=${genie_version} \
  ${srcdir}/..
docker push ${IMAGE}
