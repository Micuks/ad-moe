import pymysql
import argparse

config = {
    "host": ("The hostname to OceanBase", "localhost"),
    "port": ("The port number to OceanBase", 12881),
    "dbname": ("Database name", "test"),
    "user": ("user of the database", "root@test"),
    "password": ("the password", ""),
}


def build_index(table_name):
    conn = pymysql.connect(
        database=config["dbname"][1],
        user=config["user"][1],
        password=config["password"][1],
        host=config["host"][1],
        port=config["port"][1],
    )
    cursor = conn.cursor()
    cursor.execute(
        "CREATE INDEX index_" + table_name + "_id ON " + table_name + "(id);"
    )
    conn.commit()
    conn.close()
    return


def drop_index(table_name):
    conn = pymysql.connect(
        database=config["dbname"][1],
        user=config["user"][1],
        password=config["password"][1],
        host=config["host"][1],
        port=config["port"][1],
    )
    cursor = conn.cursor()
    drop_sql = f"alter {table_name} drop index index_{table_name}_id;"
    print(drop_sql)
    cursor.execute(drop_sql)
    # cursor.execute("DROP INDEX index_" + table_name + "_id;")

    conn.commit()
    conn.close()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", help="0 to drop, else to build", type=int)
    parser.add_argument("--table_name", type=str, default="aa", help="name of table")
    args = parser.parse_args()
    c = args.c
    table_name = args.table_name
    if c == 0:
        drop_index(table_name)
    else:
        build_index(table_name)
