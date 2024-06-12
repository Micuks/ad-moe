#!/bin/bash/

bash start_dool_big.sh
for ncolumns in 5 10 15 20; do
    for colsize in 20 40 60 80; do
        python createt.py --ncolumns $ncolumns --colsize $colsize
        for bench in tpcc tatp voter smallbank; do
            echo -e "benchmark ${bench} start $(date +%Y-%m-%d\ %H:%M:%S)"
            $OLTPBENCH_HOME/oltpbenchmark -b $bench -c ./config/oceanbase/ob${bench}_config.xml --execute=true -s 15 -d /mnt/data1/wuql/oceanbase/data/DBPA/reproduction/codegen/../logs/ &
            for nclient in 50 100 150; do
                for expid in 1 2 3; do
                    python main.py --ncolumns $ncolumns --colsize $colsize --nclient_5 $nclient --nrows 1
                done
            done
            ps -ef | grep oltpbench | awk '$0 !~ /grep/ {print $2}' | xargs kill -9
            sleep 3
            rm -r -f results
            echo -e "benchmark ${bench} end"
        done
    done
done
sleep 900
bash stop_dool_big.sh
