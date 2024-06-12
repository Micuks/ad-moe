# python dooba.py -h ODP地址 -u root@sys#集群名 -P ODP端口 -p密码
# obclient -h127.0.0.1 -P2881 -uroot@sys -Doceanbase -A

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

tmux new-session -d -s $sess1
tmux send-keys 'dool  -a -s -i -r --aio --fs --ipc --lock --raw --socket --tcp --udp --unix --vm --vm-adv --zones  --output ./metricsx.csv --noupdate 5 ' C-m
tmux new-session -d -s $sess2
tmux send-keys 'dool --postgresql-time --postgresql-all --output ./metricsy.csv --noupdate 5 > /dev/null' C-m

# tmux send-keys 'python2 dooba.py -h 127.0.0.1 -u root@sys -P 2881 -d'

