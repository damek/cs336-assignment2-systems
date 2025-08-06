#!/usr/bin/env bash
set -euo pipefail

# OUT_CSV="../outputs/csv/$(date +%F)_table1.1.2_memory.csv"
# mkdir -p "$(dirname "$OUT_CSV")"

sizes=("2.7B")
d_models=(2560)
d_ffs=(10240)
layers=(32)
heads=(32)

contexts=(128 256 512)

COMMON_STATIC="--vocab_size 10000 --rope_theta 10000 --batch_size 4 \
               --num_benchmark 1 --num_warmup 5 --memory"

# echo "timestamp,num_layers,num_heads,d_model,d_ff,context_length,batch_size,only_forward,bfloat16,mean_s,std_s,oom" > "${OUT_CSV}"

for i in "${!sizes[@]}"; do
  SIZE=${sizes[$i]}
  echo; echo "========== ${SIZE^^} MODEL =========="

  for ctx in "${contexts[@]}"; do
    for fwd_only in "no" "yes"; do
      [[ "$fwd_only" == "yes" ]] && extra_fwd="--only_forward" || extra_fwd=""

      for bfloat16 in "no" "yes"; do
        [[ "$bfloat16" == "yes" ]] && extra_bfloat16="--bfloat16" || extra_bfloat16=""
        echo "--- ctx=${ctx}  forward=${fwd_only}  bfloat16=${bfloat16}"

        python benchmarking_script.py \
            --num_layers  "${layers[$i]}"  \
            --num_heads   "${heads[$i]}"   \
            --d_model     "${d_models[$i]}"\
            --d_ff        "${d_ffs[$i]}"   \
            --context_length "${ctx}"      \
            ${extra_bfloat16}                    \
            ${extra_fwd}                   \
            ${COMMON_STATIC}
      done
    done
  done
done

echo; echo "All runs finished — results in outputs/memory"
