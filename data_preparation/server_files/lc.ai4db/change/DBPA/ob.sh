# 设置MySQL数据库的用户，如果DOOL_MYSQL_USER未设置，则使用当前用户
export DOOL_OB_USER="root"

# 设置MySQL数据库的密码
export DOOL_OB_PWD="123456"

# 设置MySQL数据库的主机地址
export DOOL_OB_HOST="localhost"

# 设置MySQL数据库的端口，MySQL默认端口通常是3306
export DOOL_OB_PORT=2881

export DOOL_OB_DB="oceanbase"

dool --ob-all --output ./metrics_ob.csv > /dev/null
