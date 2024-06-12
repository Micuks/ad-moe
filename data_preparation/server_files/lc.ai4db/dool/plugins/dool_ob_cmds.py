### Author: <lefred@inuits.be>
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

class dool_plugin(dool):
    """
    Plugin for MySQL 5 commands.
    """
    def __init__(self):
        self.name = 'ob cmds'
        self.nick = ('sel', 'ins','upd','del')
        self.vars = ('Com_select', 'Com_insert','Com_update','Com_delete')
        self.type = 'd'
        self.width = 5
        self.scale = 1

    def check(self): 
        global MySQLdb
        import pymysql
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
            self.db = pymysql.connect(**args)
        except:
            raise Exception('Cannot interface with MySQL server')

    def extract(self):
        try:
            c = self.db.cursor()
            for name in self.vars:
                c.execute("""show global status like '%s';""" % name)
                line = c.fetchone()
                if line[0] in self.vars:
                    if line[0] + 'raw' in self.set2:
                        self.set2[line[0]] = int(line[1]) - self.set2[line[0] + 'raw']
                    self.set2[line[0] + 'raw'] = int(line[1])

            for name in self.vars:
                self.val[name] = self.set2[name] * 1.0 / elapsed

            if step == op.delay:
                self.set1.update(self.set2)

        except Exception as e:
            for name in self.vars:
                self.val[name] = -1
