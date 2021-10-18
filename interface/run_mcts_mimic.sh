#!/bin/bash
launch_time="2021-05-25"
cpuct=0.001
play=200
game=flappybird
disentangler=CMONET
log_dir="${game}-output-single-cput${cpuct}-play${play}-${disentangler}-${launch_time}.out"
touch $log_dir
for ((n=0;n<51;n++))
do
    echo shell running round $n
    python3 run_mimic_learner.py -c $cpuct -p $play -r $n -d $log_dir -m mcts -g $game -n $disentangler -l $launch_time 2>&1 &
    process_id=$!
    wait $process_id
    echo shell finish running round $n
done