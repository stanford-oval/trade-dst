#!/bin/bash

srcdir=`dirname $0`
. "${srcdir}/config"
. "${srcdir}/lib.sh"

parse_args "$0" "experiment dataset model" "$@"
shift $n
if [ "$GPU" = "1" ]; then
	GPU_TYPE="p3.2xlarge"
else
	GPU_TYPE="p3.8xlarge"
fi
check_config "IAM_ROLE OWNER DATASET_OWNER IMAGE GPU GPU_TYPE"

JOB_NAME=${OWNER}-train-${experiment}-${model}
cmdline="--owner ${OWNER} --dataset_owner ${DATASET_OWNER} --experiment $experiment --dataset $dataset --model $model -- "$(requote "$@")

set -e
set -x
replace_config "${srcdir}/train.yaml.in" > "${srcdir}/train.yaml"

kubectl -n research delete job ${JOB_NAME} || true
kubectl apply -o yaml -f "${srcdir}/train.yaml"
