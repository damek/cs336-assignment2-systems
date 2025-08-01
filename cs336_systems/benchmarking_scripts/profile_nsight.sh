#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="../outputs/nsys"
mkdir -p "$OUT_DIR"

sizes=(small medium large xl "2.7B")
d_models=(768 1024 1280 1600 2560)
d_ffs=(3072 4096 5120 6400 10240)
layers=(12 24 36 48 32)
heads=(12 16 20 25 32)

contexts=(128 256 512 1024)

COMMON_STATIC="--vocab_size 10000 --rope_theta 10000 --batch_size 4 --nvtx"

for i in "${!sizes[@]}"; do
  SIZE=${sizes[$i]}
  echo; echo "========== ${SIZE^^} MODEL =========="

  for ctx in "${contexts[@]}"; do
    tag="${sizes[$i]}_ctx${ctx}"
    echo "profiling $tag"
    nsys profile --pytorch \
        --python-backtrace=cuda \
        -o "${OUT_DIR}/${tag}" \
        uv run benchmarking_script.py \
        --num_layers  "${layers[$i]}"  \
        --num_heads   "${heads[$i]}"   \
        --d_model     "${d_models[$i]}"\
        --d_ff        "${d_ffs[$i]}"   \
        --context_length "${ctx}"      \
        ${COMMON_STATIC}
  done
done

echo; echo "All runs finished — results in ${OUT_DIR}"

tar czf "${OUT_DIR}/nsys_$(date +%F).tgz" -C "$(dirname $OUT_DIR)" "$(basename $OUT_DIR)"
echo "Archive written to ${OUT_DIR}/nsys_$(date +%F).tgz"
