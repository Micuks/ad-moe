#!/bin/bash/

normal_values=(16GB)
bad_values=(512MB 128MB 32MB 8MB)
for ((i = 0; i < 4; i++)); do
    python set_knob.py -db mysql -kn memory_limit -kv ${bad_values[i]}
docker exec -it ob_32_128
            /bin/bash -c
            "obd cluster restart obcluster; sleep 1200;"
    bash start_dool_big.sh
    for bench in tpcc tatp voter smallbank; do
        echo -e "benchmark ${bench} start $(date +%Y-%m-%d\ %H:%M:%S)"
        $OLTPBENCH_HOME/oltpbenchmark -b $bench -c ./expconfig/oceanbase/ob${bench}_config_setknob.xml --execute=true -s 15 -d /mnt/data1/wuql/oceanbase/data/DBPA/reproduction/codegen/../logs/ &
        ps -ef | grep oltpbench | awk '$0 !~ /grep/ {print $2}' | xargs kill -9
        sleep 3
        rm -r -f results
        echo -e "benchmark ${bench} end"
    done
    sleep 900
    bash stop_dool_big.sh
done
python set_knob.py -db mysql -kn memory_limit -kv ${normal_values[0]}
docker exec -it ob_32_128
            /bin/bash -c
            "obd cluster restart obcluster; sleep 1200;"
