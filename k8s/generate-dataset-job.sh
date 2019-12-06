#!/bin/bash

srcdir=`dirname $0`
. "${srcdir}/lib.sh"

parse_args "$0" "owner dataset_owner experiment dataset" "$@"
shift $n

set -e
set -x

mkdir workdir
cd workdir

pwd
aws s3 sync s3://almond-research/${owner}/workdir-${experiment}/ .

export GENIE_TOKENIZER_ADDRESS=tokenizer.default.svc.cluster.local:8888
export TZ=America/Los_Angeles


make "experiment=${experiment}" "owner=${dataset_owner}" "tradedir=${srcdir}" "geniedir=/opt/genie-toolkit" "$@" data-generated
aws s3 sync data-generated/ "s3://almond-research/${dataset_owner}/dataset/${experiment}/${dataset}/"