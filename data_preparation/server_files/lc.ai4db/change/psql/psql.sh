# make clean
# make 
# make install 

export DSTAT_PG_USER=puser
export DSTAT_PG_PWD="123456"
export DSTAT_PG_HOST=localhost
export DSTAT_PG_PORT=5432
export DOOL_OB_DB="test"

# dool -a -s -i -r --aio --fs --ipc --lock --raw --socket --tcp --udp --unix --vm --vm-adv --zones --postgresql-conn --postgresql-lockwaits --postgresql-settings --output ./metricsx.csv --noupdate 5 > /dev/null
# dool -a -s -i -r --aio --fs --ipc --lock --raw --socket --tcp --udp --unix --vm --vm-adv --zones --postgresql-conn --postgresql-lockwaits --output ./metricsx.csv --noupdate 5 > /dev/null
dool --postgresql-conn --postgresql-lockwaits --postgresql-time --postgresql-all --postgresql-settings --output ./metrics_postgres_only.csv --noupdate 5 > /dev/null
# dool --postgresql-settings --output ./metricsx.csv --noupdate 5
# dool --postgresql-time --postgresql-all --output ./metricsy.csv --noupdate 5 > /dev/null
