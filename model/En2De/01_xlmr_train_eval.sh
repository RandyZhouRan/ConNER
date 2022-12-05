#!/usr/bin/env bash

GPU=0
RDROP_W=0.5
UDA_W=0.5
N=
DATA=En2De
MODEL=xlm-roberta-large
OUTPUT="${DATA}_${MODEL}"
OUTPUT_DIR=results

NUM=10

BS=16
ACCU_STEPS=1

for n in 42;do
  RESULTS="${OUTPUT}-run$n"
  SEED=${n}
  CUDA_VISIBLE_DEVICES=$GPU python ../../trainer.py \
      --batch_size $BS \
      --device=cuda:0 \
      --seed $SEED \
      --dataset="$DATA" \
      --model_folder="$RESULTS" \
      --embedder_type="$MODEL" \
      --parallel_embedder 1 \
      --num_epochs "$NUM" \
      --rdrop_weight "$RDROP_W" \
      --uda_weight "$UDA_W" | tee "${OUTPUT}".log_run${n}.txt
done
