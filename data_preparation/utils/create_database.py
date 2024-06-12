import pymysql
import argparse


config = {
    "host": ("The hostname to oceanbase", "localhost"),
    "port": ("The port number to oceanbase", 12881),
    "dbname": ("Database name", "test"),
    "user": ("user of the database", "root@test"),
    "password": ("the password", ""),
}


def create_database():
    conn = pymysql.connect(
        user=config["user"][1],
        password=config["password"][1],
        host=config["host"][1],
        port=config["port"][1],
    )

    cursor = conn.cursor()
    cursor.execute("set global autocommit=1")
    for dbname in ["tpcc", "tatp", "smallbank", "voter", "tpcc_test"]:
        cmd = "create database " + dbname + ";"
        cursor.execute(cmd)
    conn.commit()
    return


if __name__ == "__main__":
    create_database()
