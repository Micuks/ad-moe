#!/bin/bash

# 设置MySQL数据库的用户，如果DOOL_MYSQL_USER未设置，则使用当前用户
export DOOL_MYSQL_USER="root"

# 设置MySQL数据库的密码
export DOOL_MYSQL_PWD="123456"

# 设置MySQL数据库的主机地址
export DOOL_MYSQL_HOST="localhost"

# 设置MySQL数据库的端口，MySQL默认端口通常是3306
export DOOL_MYSQL_PORT=2881

# 设置MySQL数据库的Socket文件路径，如果有的话
# export DOOL_MYSQL_SOCKET=""

# python3 test_conn.py
dool --mysql5-cmds --mysql5-conn --mysql5-innodb --mysql5-io --mysql5-keys 

# OceanBase 连接启动
# obclient -h127.0.0.1 -P2881 -uroot@sys -Doceanbase -A


git push -u origin "who" To https://gitee.com/lcccccx/ai4db_lc.git