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
best_model=$(ls -d save/TRADE*/HDD*BSZ* | sort -r | head -n1)

echo "Everything" > results
python3 myTest.py -path "$best_model" "$@" | tee -a results
aws s3 cp prediction_* s3://almond-research/${owner}/models/${experiment}/${model}/predictions/full/
aws s3 rm prediction_*

for d in hotel train restaurant attraction taxi ; do
  echo "Only" $d >> results
  python3 myTest.py -path "$best_model" -onlyd "$d" "$@" | tee -a results
  aws s3 cp prediction_* s3://almond-research/${owner}/models/${experiment}/${model}/predictions/${d}/
  aws s3 rm prediction_*
done

aws s3 cp results s3://almond-research/${owner}/models/${experiment}/${model}/results

