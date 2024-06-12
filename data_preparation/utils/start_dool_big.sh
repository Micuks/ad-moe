#!/bin/bash
docker_container="labob_4_16"
server_files_dir="/data/DBPA/reproduction/server_files"

docker exec -it ${docker_container} /bin/bash -c "cd ${server_files_dir} && bash ./start_dool2.sh; \
echo -e \"server_dool_begin\t$(date +%Y-%m-%d\ %H:%M:%S);\""

# ssh user@localhost 'bash -s' <<'ENDSSH'
# cd /home/user && bash start_dool2.sh
# echo -e "server_dool_begin\t$(date +%Y-%m-%d\ %H:%M:%S)"
# ENDSSH

echo -e "client_dool_begin\t$(date +%Y-%m-%d\ %H:%M:%S)"
