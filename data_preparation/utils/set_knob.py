import pymysql
import argparse


config = {
    "host": ("The hostname to oceanbase", "localhost"),
    "port": ("The port number to oceanbase", 12881),
    "dbname": ("Database name", "oceanbase"),
    "user": ("user of the database", "root@sys"),
    "password": ("the password", "hoVNpg8CNXM9bZdqTUaL"),
}


def set_knob(knob_name, knob_value):
    conn = pymysql.connect(
        host=config["host"][1],
        port=config["port"][1],
        user=config["user"][1],
        password=config["password"][1],
        db=config["dbname"][1],
        charset="utf8",
    )

    cursor = conn.cursor()

    cursor.execute("set global autocommit=1")

    cmd = f"alter system set {knob_name}='{knob_value}';"
    # cmd = "SET GLOBAL " + knob_name + " = " + knob_value + ";"
    cursor.execute(cmd)
    conn.commit()

    conn.close()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-kn", help="knob name")
    parser.add_argument("-kv", help="knob value")
    args = parser.parse_args()
    knob_name = args.kn
    knob_value = args.kv
    set_knob(knob_name, knob_value)
