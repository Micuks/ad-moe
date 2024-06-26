global pg_user
pg_user = os.getenv('DSTAT_PG_USER') or os.getenv('USER')

global pg_pwd
pg_pwd = os.getenv('DSTAT_PG_PWD')

global pg_host
pg_host = os.getenv('DSTAT_PG_HOST')

global pg_port
pg_port = os.getenv('DSTAT_PG_PORT')

class dool_plugin(dool):
    """
    Plugin for PostgreSQL connections.
    """

    def __init__(self):
        self.name = 'postgresql settings'
        self.nick = ('shared_buffers',
                     'work_mem', 'bgwriter_delay',
                     'max_connections',
                     'autovacuum_work_mem',
                     'temp_buffers', 'autovacuum_max_workers',
                     'maintenance_work_mem', 'checkpoint_timeout',
                     'max_wal_size', 'checkpoint_completion_target',
                     'wal_keep_segments', 'wal_segment_size')
        self.vars = self.nick
        self.type = 'f'
        self.width = 9
        self.scale = 1

    def check(self):
        global psycopg2
        import psycopg2
        try:
            args = {}
            if pg_user:
                args['user'] = pg_user
            if pg_pwd:
                args['password'] = pg_pwd
            if pg_host:
                args['host'] = pg_host
            if pg_port:
                args['port'] = pg_port

            self.db = psycopg2.connect(**args)
        except Exception as e:
            raise Exception('Cannot interface with PostgreSQL server, %s' % e)

    def extract(self):
        # import PostgresqlConn
        try:
            # with PostgresqlConn() as c:
            c = self.db.cursor()
            sql = 'select name, setting from pg_settings where name in {}; '
            sql = sql.format(self.vars)
            c.execute(sql)
            res = c.fetchall()

            for k, v in res:
                v = float(v)
                self.val[k] = v

        except Exception as e:
            for name in self.vars:
                self.val[name] = -1

# vim:ts=4:sw=4:et
