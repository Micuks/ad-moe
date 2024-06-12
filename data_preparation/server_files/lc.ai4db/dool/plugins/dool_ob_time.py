import os
import pymysql

global ob_user
ob_user = os.getenv("DOOL_OB_USER") or os.getenv("USER")

global ob_pwd
ob_pwd = os.getenv("DOOL_OB_PWD")

global ob_host
ob_host = os.getenv("DOOL_OB_HOST")

global ob_port
ob_port = os.getenv("DOOL_OB_PORT")

global ob_db
ob_db = os.getenv("DOOL_OB_DB")

print(ob_user, ob_pwd, ob_host, ob_port, ob_db, sep="\n")


class dool_plugin(dool):
    """Plugin to collect timestamp for OceanBase

    Args:
        dool (_type_): _description_
    """

    def __init__(self):
        self.name = "ob time"
        self.nick = ("time",)
        self.vars = ("Time",)
        self.type = "d"
        self.width = 10
        self.scale = 1

    def check(self):
        import pymysql

        try:
            args = {}
            if ob_user:
                args["user"] = ob_user
            if ob_pwd:
                args["passwd"] = ob_pwd
            if ob_host:
                args["host"] = ob_host
            if ob_port:
                args["port"] = int(ob_port)
            if ob_db:
                args["db"] = ob_db
            self.db = pymysql.connect(**args)
        except:
            raise Exception("Cannot interface with OceanBase server")

    def extract(self):
        try:
            c = self.db.cursor()
            sql = f"select ceil(unix_timestamp(now()))"
            c.execute(sql)
            Time = c.fetchone()
            print("Time: ", Time)
            self.val[self.vars[0]] = Time[0]

        except Exception as e:
            print(e)
            for name in self.vars:
                self.val[name] = -1
