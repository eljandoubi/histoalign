#!/bin/sh
# The runs behind results/: three splits, a few variants, three seeds each. About an hour on a laptop.
# Features must be cached first:  histoalign features data/raw/<sample> ... --out data/features/lunit
set -e
A1=V1_Breast_Cancer_Block_A_Section_1
A2=V1_Breast_Cancer_Block_A_Section_2
FEATURES=data/features/lunit
mkdir -p results

run() {
  split=$1; variant=$2; shift 2
  for seed in 0 1 2; do
    out=results/${split}__${variant}__seed${seed}.json
    [ -f "$out" ] && continue
    uv run histoalign run --features $FEATURES "$@" --seed $seed --out "$out" > /dev/null
    echo "$out"
  done
}

for variant_args in "default:" "nbr2:--neighbour-radius 2" "indrop:--input-dropout 0.5" "sigmoid:--loss sigmoid"; do
  variant=${variant_args%%:*}
  extra=${variant_args#*:}
  run a1_block "$variant" --block $A1 $extra
  run a1_to_a2 "$variant" --train $A1 --test $A2 $extra
  run a2_to_a1 "$variant" --train $A2 --test $A1 $extra
done
