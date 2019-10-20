#!/bin/bash

srcdir=`dirname $0`
. "${srcdir}/lib.sh"

parse_args "$0" "owner dataset_owner experiment dataset model" "$@"
shift $n

set -e
set -x

#aws s3 sync s3://almond-research/${dataset_owner}/dataset/${experiment}/${dataset}/ data/

python3 myTrain.py "$@"

aws s3 sync save/*/ s3://almond-research/${owner}/models/${experiment}/${model}/
