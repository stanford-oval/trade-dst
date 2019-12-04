#!/bin/bash

srcdir=`dirname $0`
. "${srcdir}/config"
. "${srcdir}/lib.sh"

parse_args "$0" "experiment dataset model" "$@"
shift $n
check_config "IAM_ROLE OWNER DATASET_OWNER IMAGE GPU"

JOB_NAME=${OWNER}-train-${experiment}-${model}
cmdline="--owner ${OWNER} --dataset_owner ${DATASET_OWNER} --experiment $experiment --dataset $dataset --model $model -- "$(requote "$@")

set -e
set -x
replace_config "${srcdir}/train.yaml.in" > "${srcdir}/train.yaml"

kubectl -n research delete job ${JOB_NAME} || true
kubectl apply -o yaml -f "${srcdir}/train.yaml"
