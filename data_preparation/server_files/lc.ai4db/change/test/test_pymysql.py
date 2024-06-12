#!/usr/bin/python3
import os
import pymysql
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

global mysql_db
mysql_db = os.getenv('DOOL_OB_DB')
try:
    args = {}
    if mysql_user:
        args['user'] = mysql_user
    if mysql_pwd:
        args['passwd'] = ""
    if mysql_host:
        args['host'] = mysql_host
    if mysql_port:
        args['port'] = int(mysql_port)
    if mysql_socket:
        args['unix_socket'] = mysql_socket
    if mysql_db:
        args['db'] = mysql_db
    conn = pymysql.connect(**args)
except:
    raise Exception('Cannot interface with MySQL server')
cur = conn.cursor()

try:

    # #创建表 cities
    # sql = 'create table cities (id int, name varchar(24))'
    # cur.execute(sql)

    #往 cities 表中插入两组数据
    sql = "insert into cities values(1,'hangzhou'),(2,'shanghai')"
    cur.execute(sql)

    #查询 cities 表中的所有数据
    sql = 'select * from cities;'
    cur.execute(sql)

    sql = "show global status like 'Com_select';"
    cur.execute(sql)
    ans = cur.fetchall()
    print(ans)

    # #删除表 cities
    # sql = 'drop table cities'
    # cur.execute(sql)

finally:
    cur.close()
    conn.close()
