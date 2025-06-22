#!/bin/bash
set -e


cd "$(dirname "${BASH_SOURCE[0]}")"


cargo build --bin ahc045-a
cargo build --release --bin ahc045-a

cd tools


#../../target/release/tester ../../target/debug/ahc045-a < in/0000.txt > out/0000.txt
../../target/release/tester ../../target/release/ahc045-a < in/0000.txt > out/0000.txt


# for i in $(seq 0 0)
# do
#     echo "--------------------------------"
#     num=$(printf "%04d" $i)
#     echo "case: $num"

#     # 名前付きパイプの作成
#     IN_FIFO_NAME="ahc045_test_in_$num"
#     mkfifo "$IN_FIFO_NAME"

    
#     OUT_FIFO_NAME="ahc045_test_out_$num"
#     mkfifo "$OUT_FIFO_NAME"

    
#     trap "rm -f $IN_FIFO_NAME; rm -f $OUT_FIFO_NAME" EXIT
    

#     ../target/release/ahc045-interactor > "$IN_FIFO_NAME" < "$OUT_FIFO_NAME" &
#     ../target/release/ahc045-a < "$IN_FIFO_NAME" > "$OUT_FIFO_NAME" &
    
#     cat "in/$num.txt" > "$OUT_FIFO_NAME"
    
#     #../target/release/ahc045-a < in/${num}.txt > out/out_${num}.txt
#     #time=$(cat out/out_${num}.txt | grep "#time:" | awk '{print $2}')
#     #echo "time: $time"
#     # sleep 0.3

# done

# # すべてのプロセスの完了を待つ
# wait
