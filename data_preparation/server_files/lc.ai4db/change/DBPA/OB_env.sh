#!/bin/bash

# 设置MySQL数据库的用户，如果DOOL_OB_USER未设置，则使用当前用户
export DOOL_OB_USER="root"

# 设置MySQL数据库的密码
export DOOL_OB_PWD="hoVNpg8CNXM9bZdqTUaL"

# 设置MySQL数据库的主机地址
export DOOL_OB_HOST="127.0.0.1"

# 设置MySQL数据库的端口，MySQL默认端口通常是3306
export DOOL_OB_PORT=12883

export DOOL_OB_DB="oceanbase"

# 设置MySQL数据库的Socket文件路径，如果有的话
# export DOOL_OB_SOCKET=""

# python3 test_conn.py
# dool --mysql5-cmds --mysql5-conn --mysql5-innodb --mysql5-io --mysql5-keys

# OceanBase 连接启动
# obclient -h127.0.0.1 -P2881 -uroot@sys -Doceanbase -A
# mysql --connect_timeout=5 -s -N -h127.0.0.1 -P12883 -uroot -p'hoVNpg8CNXM9bZdqTUaL' -Doceanbase -A
# python3 test_pymysql.py

dool --ob-all --output ./metrics_ob.csv >/dev/null
