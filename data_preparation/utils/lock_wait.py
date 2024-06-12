import time
from multiprocessing.pool import *
import pymysql
import random
import string
import os
import csv
import argparse
import subprocess

config = {
    "host": ("The hostname to OceanBase", "localhost"),
    "port": ("The port number to OceanBase", 12881),
    "dbname": ("Database name", "test"),
    "user": ("user of the database", "root@test"),
    "password": ("the password", ""),
}


def getTime():
    status, tim = subprocess.getstatusoutput("date +'%F %H:%M:%S'")
    tims = tim.split(" ")
    tims[1] = tims[1].replace("-", ":")
    return tims[0] + " " + tims[1]


def fault_time(fault_id, wait_type):
    print("fault_{}_{}\t{}".format(fault_id, wait_type, getTime()))


def update_all(database_name, table_name, column_name, duration, interval):
    try:
        conn = pymysql.connect(
            database=database_name,
            user=config["user"][1],
            password=config["password"][1],
            host=config["host"][1],
            port=config["port"][1],
        )
        cursor = conn.cursor()
        print(conn.autocommit)
        xx = 11
        sql = "update {} set {}='{}';".format(table_name, column_name, xx)
        fault_time("lockwait", "begin1")
        cursor.execute(sql)
        fault_time("lockwait", "begin2")
        cwd = os.getcwd()
        os.system(
            f"$OLTPBENCH_HOME/oltpbenchmark -b {database_name} -c {cwd}/config/oceanbase/ob{database_name}_config.xml --execute=true -s 15 -o outputfile &"
        )
        time.sleep(duration)
        conn.rollback()
        fault_time("lockwait", "end1")
        time.sleep(interval)
        fault_time("lockwait", "end2")
        conn.close()
    except Exception as e:
        fault_time("lockwait", "exception")
        print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=60, help="duration")
    parser.add_argument("--interval", type=int, default=30, help="interval")
    parser.add_argument(
        "--database_name", type=str, help="name of the database to update"
    )
    parser.add_argument("--table_name", type=str, help="name of the table to update")
    parser.add_argument("--column_name", type=str, help="name of the column to update")

    args = parser.parse_args()
    duration = args.duration
    interval = args.interval
    database_name = args.database_name
    table_name = args.table_name
    column_name = args.column_name

    random.seed(time.time())

    print("update tabel {}".format(table_name))
    update_all(database_name, table_name, column_name, duration, interval)
