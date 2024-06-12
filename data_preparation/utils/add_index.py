import pymysql
import argparse

config = {
    "host": ("The hostname to OceanBase", "localhost"),
    "port": ("The port number to OceanBase", 12881),
    "dbname": ("Database name", "test"),
    "user": ("user of the database", "root@test"),
    "password": ("the password", ""),
}


def build_index(table_name, idx_num):
    conn = pymysql.connect(
        database=config["dbname"][1],
        user=config["user"][1],
        password=config["password"][1],
        host=config["host"][1],
        port=config["port"][1],
    )
    cursor = conn.cursor()
    for i in range(0, idx_num):
        the_sql = (
            "CREATE INDEX index_"
            + table_name
            + "_"
            + str(i)
            + " ON "
            + table_name
            + "(name"
            + str(i)
            + ");"
        )
        print(the_sql)
        cursor.execute(the_sql)
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

    # Adjusted for MySQL-syntax
    show_index_query = f"show index from `{table_name}` where Key_name != 'PRIMARY';"
    cursor.execute(show_index_query)
    indices = cursor.fetchall()

    for idx in indices:
        index_name = idx[2]
        drop_sql = f"alter table `{table_name}` drop index `{index_name}`;"
        print(drop_sql)
        cursor.execute(drop_sql)

    conn.commit()
    conn.close()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", help="number of columns for index, 0 to drop all", type=int
    )
    parser.add_argument(
        "--table_name", help="table to create the index on", type=str, default="aa"
    )
    args = parser.parse_args()
    c = args.c
    table_name = args.table_name

    if c == 0:
        drop_index(table_name)
    else:
        build_index(table_name, c)
