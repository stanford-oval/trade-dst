#!/bin/bash

srcdir=`dirname $0`
. "${srcdir}/config"
. "${srcdir}/lib.sh"

parse_args "$0" "experiment dataset" "$@"
shift $n
check_config "IAM_ROLE OWNER IMAGE"

JOB_NAME=${OWNER}-gen-dataset-${experiment}-${dataset}
cmdline="--owner ${OWNER} --dataset_owner ${DATASET_OWNER} --experiment $experiment --dataset $dataset -- "$(requote "$@")

set -e
set -x
replace_config "${srcdir}/generate-dataset.yaml.in" > "${srcdir}/generate-dataset.yaml"

kubectl -n research delete job ${JOB_NAME} || true
kubectl apply -o yaml -f "${srcdir}/generate-dataset.yaml"
