#!/bin/bash

# I/O saturation due to other processes. It is a representative of
# resource bottleneck and differs from the anomaly caused by the workload.
docker_container="labob_4_16"

export OLTPBENCH_HOME="${OLTPBENCH_HOME:-/mnt/data1/wuql/oceanbase/data/DBPA/oltpbench}"
export DBPA_UTILS_DIR="${DBPA_UTILS_DIR:-/home/wuql/disk/oceanbase/data/DBPA/reproduction/utils}"
export IN_CONTAINER_SERVER_FILES_DIR="/data/DBPA/reproduction/server_files"

bash ${DBPA_UTILS_DIR}/start_dool_big.sh
cd ${OLTPBENCH_HOME}
# for bench in tpcc tatp voter smallbank; do
for bench in tatp voter smallbank; do
    echo -e "benchmark ${bench} start"
    $OLTPBENCH_HOME/oltpbenchmark -b $bench -c ${DBPA_UTILS_DIR}/../config/oceanbase/ob${bench}_config.xml --execute=true -s 15 -o outputfile &
    docker exec ${docker_container} /bin/bash -c "bash ${IN_CONTAINER_SERVER_FILES_DIR}/io_saturation_server.sh"
    ps -ef | grep oltpbench | awk '$0 !~ /grep/ {print $2}' | xargs kill -9
    sleep 3
    rm -r -f results
    echo -e "benchmark ${bench} end"
done
sleep 3600
bash ${DBPA_UTILS_DIR}/stop_dool_big.sh
