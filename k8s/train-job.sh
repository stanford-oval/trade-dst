#!/bin/bash

srcdir=`dirname $0`
. "${srcdir}/lib.sh"

parse_args "$0" "owner task_name experiment dataset model" "$@"
shift $n

set -e
set -x

aws s3 sync s3://almond-research/${owner}/dataset/${experiment}/${dataset} data/

mkdir -p ./model

decanlp train \
  --path ./model \
  "$@"

aws s3 sync .model/ s3://almond-research/${owner}/models/${experiment}/${model}
