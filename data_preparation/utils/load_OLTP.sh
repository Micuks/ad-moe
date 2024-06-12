#!/bin/bash
# Usage: cd .. && ./utils/load_OLTP.sh
# Because this script requires working-directory

# Env variables
export OLTPBENCH_HOME="${OLTPBENCH_HOME:-'/mnt/data1/wuql/oceanbase/data/DBPA/oltpbench'}"
export DBPA_UTILS_DIR="${DBPA_UTILS_DIR:-/home/wuql/disk/oceanbase/data/DBPA/reproduction/utils}"
export LOGS_DIR="${WORKING_DIR}/logs/"
mkdir -p ${LOGS_DIR}
echo -e "working directory: ${WORKING_DIR}"
echo -e "logs directory: ${LOGS_DIR}"

cd ${OLTPBENCH_HOME}
for bench in tpcc tatp smallbank voter; do
    echo -e "===benchmark ${bench} start==="
    $OLTPBENCH_HOME/oltpbenchmark -b $bench -c ${DBPA_UTILS_DIR}/../config/oceanbase/ob${bench}_config.xml --create=true --load=true -s 15 -o outputfile
    pid_oltpbench=$(ps -ef | grep "oltpbench" | awk '$0 !~ /grep/ {print $2}')
    echo "Killing $pid_oltpbench..."
    $pid_oltpbench | xargs kill -9
    sleep 3
    rm -r -f results
    echo -e "===benchmark ${bench} end==="
    sleep 3
done
# back
cd ${WORKING_DIR}
