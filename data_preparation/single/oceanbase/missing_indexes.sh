#!/bin/bash/

bash start_dool_big.sh
for ncolumns in 5 10 20; do
    for colsize in 50 100; do
        for nrows in 2000000 4000000; do
            python createt.py --ncolumns $ncolumns --colsize $colsize --nrows $nrows
            for bench in tpcc tatp voter smallbank; do
                echo -e "benchmark ${bench} start $(date +%Y-%m-%d\ %H:%M:%S)"
                $OLTPBENCH_HOME/oltpbenchmark -b $bench -c ./config/oceanbase/ob${bench}_config.xml --execute=true -s 15 -d /mnt/data1/wuql/oceanbase/data/DBPA/reproduction/codegen/../logs/ &
                for nclient in 5 10; do
                    for expid in 1 2 3; do
                        python main.py --ncolumns $ncolumns --colsize $colsize --client_2 $nclient --tabsize $nrows
                    done
                done
                ps -ef | grep oltpbench | awk '$0 !~ /grep/ {print $2}' | xargs kill -9
                sleep 3
                rm -r -f results
                echo -e "benchmark ${bench} end"
            done
        done
    done
done
sleep 900
bash stop_dool_big.sh
