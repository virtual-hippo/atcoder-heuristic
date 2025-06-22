#!/bin/bash
set -e


score_sum=0
time_sum=0
count=0

# 並列実行して一時ファイルに結果を保存
cargo build --bin ahc046-a
cargo build --release --bin ahc046-a

for i in $(seq 0 0)
do
    echo "--------------------------------"
    num=$(printf "%04d" $i)
    echo "case: $num"
    ../target/release/ahc046-a < in/${num}.txt > out/${num}.txt 2> out/${num}_err.txt
    time=$(cat out/${num}.txt | grep "#time:" | awk '{print $2}')
    echo "time: $time"
    # sleep 0.3
done

# すべてのプロセスの完了を待つ
wait

# 結果の集計
# for i in $(seq 0 49)
# do
#     num=$(printf "%04d" $i)
#     time=$(cat out/out_${num}.txt | grep "#time:" | awk '{print $2}')
#     score=$(cat out/out_${num}.txt | grep "#score:" | awk '{print $2}')
    
#     time_sum=$((time_sum + time))
#     score_sum=$((score_sum + score))
#     count=$((count + 1))
    
# done

# if [ $count -gt 0 ]; then
#     echo "Total Score: $score_sum"
#     echo "Average Time: $((time_sum / count))"
# fi

