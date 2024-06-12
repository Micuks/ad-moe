#!/bin/bash

sess1=monitoring1
sess2=monitoring2
tmux has-session -t $sess1 2>/dev/null
if [ $? == 0 ]; then
    tmux kill-session -t $sess1 && echo $sess1" killed"
fi
tmux has-session -t $sess2 2>/dev/null
if [ $? == 0 ]; then
    tmux kill-session -t $sess2 && echo $sess2" killed"
fi

# For OceanBase
tmux new-session -d -s $sess1
tmux send-keys './lc.ai4db/dool/dool --ob-time -a -s -i -r --aio --fs --ipc --lock --raw --socket --tcp --udp --unix --vm --vm-adv --zones --output ./metricsx_ob_$(date +%Y-%m-%d%H_%M_%S%Z).csv --noupdate 5 > /dev/null' C-m
tmux new-session -d -s $sess2
tmux send-keys './lc.ai4db/dool/dool --ob-time --ob-all --output ./metricsy_ob_$(date +%Y-%m-%d%H_%M_%S%Z).csv --noupdate 5 > /dev/null' C-m

# For PostgreSQL
# tmux new-session -d -s $sess1
# tmux send-keys './dool/dool1 --postgresql-time -a -s -i -r --aio --fs --ipc --lock --raw --socket --tcp --udp --unix --vm --vm-adv --zones --postgresql-conn --postgresql-lockwaits --postgresql-settings --output ./metricsx.csv --noupdate 5 > /dev/null' C-m
# tmux new-session -d -s $sess2
# tmux send-keys './dool/dool1 --postgresql-time --postgresql-all --output ./metricsy.csv --noupdate 5 > /dev/null' C-m
