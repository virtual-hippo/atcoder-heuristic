#!/bin/bash
set -e

for i in $(seq 0 49)
do
    num=$(printf "%04d" $i)
    cargo run --bin ahc043-a < in/${num}.txt > out/out_${num}.txt
done