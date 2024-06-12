import os
import MySQLdb
global mysql_user
mysql_user = os.getenv('DOOL_OB_USER') or os.getenv('USER')

global mysql_pwd
mysql_pwd = os.getenv('DOOL_OB_PWD')

global mysql_host
mysql_host = os.getenv('DOOL_OB_HOST')

global mysql_port
mysql_port = os.getenv('DOOL_OB_PORT')

global mysql_socket
mysql_socket = os.getenv('DOOL_OB_SOCKET')
try:
    args = {}
    if mysql_user:
        args['user'] = mysql_user
    if mysql_pwd:
        args['passwd'] = mysql_pwd
    if mysql_host:
        args['host'] = mysql_host
    if mysql_port:
        args['port'] = int(mysql_port)
    if mysql_socket:
        args['unix_socket'] = mysql_socket

    db = MySQLdb.connect(**args)
except Exception as e:
    raise Exception('Cannot interface with MySQL server, %s' % e)
cursor = db.cursor()
cursor.execute("use test");
# 执行一个插入操作
try:
    cursor.execute("INSERT INTO test_ob VALUES (35);")
    db.commit()
except MySQLdb.Error as e:
    db.rollback()  # 回滚事务
    print("Error: ", e)

# 执行一个查询
try:
    cursor.execute("SELECT * FROM test_ob")
    rows = cursor.fetchall()
    for row in rows:
        print(row)
except MySQLdb.Error as e:
    print("Error: ", e)
# 关闭游标和连接
cursor.close()
db.close()

