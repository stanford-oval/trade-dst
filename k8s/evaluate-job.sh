#!/bin/bash

srcdir=`dirname $0`
. "${srcdir}/lib.sh"

parse_args "$0" "owner dataset_owner experiment dataset model" "$@"
shift $n

set -e
set -x

aws s3 sync s3://almond-research/${dataset_owner}/dataset/${experiment}/${dataset}/ data/
aws s3 sync s3://almond-research/${owner}/models/${experiment}/${model}/ save/

ls -d save/TRADE*/ || ln -s . save/TRADE

# note: myTest is very, very broken, and assumes a very specific directory layout inside save/
best_model=$(ls -d save/TRADE*/* | sort -r | head -n1)
python3 myTest.py -path "$best_model" "$@" >
