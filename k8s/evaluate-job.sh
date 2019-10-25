#!/bin/bash

srcdir=`dirname $0`
. "${srcdir}/lib.sh"

parse_args "$0" "owner dataset_owner experiment dataset model" "$@"
shift $n

set -e
set -x

aws s3 sync s3://almond-research/${dataset_owner}/dataset/${experiment}/${dataset}/ data/
aws s3 sync s3://almond-research/${owner}/models/${experiment}/${model}/ save/
docker ps -a
ls -d save/TRADE*/ || ln -s . save/TRADE
(cd save ; ls lang-all.pkl || ln -s TRADE*/lang-all.pkl lang-all.pkl )
(cd save ; ls mem-lang-all.pkl || ln -s TRADE*/mem-lang-all.pkl mem-lang-all.pkl )

# note: myTest is very, very broken, and assumes a very specific directory layout inside save/
best_model=$(ls -d save/TRADE*/HDD*BSZ* | sort -r | head -n1)

echo "Everything" > results
python3 myTest.py -path "$best_model" "$@" | tee -a results
for pred_file in prediction_*; do
  aws s3 cp ${pred_file} s3://almond-research/${owner}/models/${experiment}/${model}/predictions/full/
done

for d in hotel train restaurant attraction taxi ; do
  echo "Only" $d >> results
  python3 myTest.py -path "$best_model" -onlyd "$d" "$@" | tee -a results
  for pred_file in prediction_*; do
    aws s3 cp ${pred_file} s3://almond-research/${owner}/models/${experiment}/${model}/predictions/${d}/
  done

done

aws s3 cp results s3://almond-research/${owner}/models/${experiment}/${model}/results

