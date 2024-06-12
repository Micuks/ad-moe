import pymysql
import time
from multiprocessing.pool import *
import random
import argparse
import numpy as np

config = {
    "host": ("The hostname to OceanBase", "localhost"),
    "port": ("The port number to OceanBase", 12881),
    "dbname": ("Database name", "test"),
    "user": ("user of the database", "root@test"),
    "password": ("the password", ""),
}


def execute_sql(sqls):
    conn = pymysql.connect(
        database=config["dbname"][1],
        user=config["user"][1],
        password=config["password"][1],
        host=config["host"][1],
        port=config["port"][1],
    )
    cursor = conn.cursor()
    for sql in sqls:
        cursor.execute(sql)
    conn.commit()
    conn.close()


def concurrent_execute_sql(client, duration, sql, max_id=0, commit_interval=500):
    pool = Pool(client)
    execute_cnt = []
    for _ in range(client):
        execute_cnt.append(
            pool.apply_async(
                execute_sql_duration, (duration, sql, max_id, commit_interval)
            )
        )
    print(
        "avg_exe_cnt : {}".format(
            np.mean([execute_cnt[i].get() for i in range(client)])
        )
    )
    pool.close()
    pool.join()


def execute_sql_duration(duration, sql, max_id=0, commit_interval=500):
    conn = pymysql.connect(
        database=config["dbname"][1],
        user=config["user"][1],
        password=config["password"][1],
        host=config["host"][1],
        port=config["port"][1],
    )
    cursor = conn.cursor()
    start = time.time()
    cnt = 0
    if duration > 0:
        while time.time() - start < duration:
            if max_id > 0:
                id = random.randint(1, max_id - 1)
                cursor.execute(sql + str(id) + ";")
            else:
                cursor.execute(sql)
            cnt += 1
            if cnt % commit_interval == 0:
                conn.commit()
    else:
        if max_id > 0:
            id = random.randint(1, max_id - 1)
            cursor.execute(sql + str(id) + ";")
        else:
            cursor.execute(sql)
        cnt += 1
        conn.commit()
    conn.commit()
    conn.close()

    return cnt
