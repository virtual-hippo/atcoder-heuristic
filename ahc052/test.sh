#!/bin/bash
set -e

score_sum=0
time_sum=0
count=0
case_num=99

# 一時ファイルに結果を保存
cargo build --release --bin ahc052-a
for i in $(seq 0 $case_num)
do
    echo "--------------------------------"
    num=$(printf "%04d" $i)
    echo "case: $num"
    ../target/release/ahc052-a < in/${num}.txt > out/${num}.txt  2> out/${num}_err.txt
    #time=$(cat out/${num}.txt | grep "#time:" | awk '{print $2}')
    #echo "time: $time"
    # sleep 0.3
done

# すべてのプロセスの完了を待つ
wait

# 結果の集計
for i in $(seq 0 $case_num)
do
    num=$(printf "%04d" $i)
    time=$(cat out/${num}_err.txt | grep "#time:" | awk '{print $2}')
    score=$(cat out/${num}_err.txt | grep "#score:" | awk '{print $2}')
    
    time_sum=$((time_sum + time))
    score_sum=$((score_sum + score))
    count=$((count + 1))
    
done

if [ $count -gt 0 ]; then
    echo "Total Score: $score_sum"
    echo "Average Time: $((time_sum / count))"
fi
