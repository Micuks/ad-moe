#!/usr/bin/env python2
# -*- mode: python;coding: utf-8 -*-

# dooba ---
#
# Filename: dooba
# Description: `dooba' is a easy tools monitoring oceanbase cluster for
#               oceanbase admins. It's based on python curses library, and is a
#               powerful tool for watching oceanbase cluster status with
#               straightfoward vision.
# Created: 2013-07-23 21:05:08 (+0800)
# Version: 2.1.3
# Last-Updated: 2018-06-07T16:57:32+0800
#           By: Shi Yudi
#     Update #: 7247
#

# Change Log:
# 2018-06-06    Shi Yudi
#    Last-Updated: 2018-06-06T19:14:36+0800 #7237 (Shi Yudi)
#    1. 修复Gallery页面切租户信息不显示的问题
#
# 2018-01-16    Shi Yudi
#    Last-Updated: 2018-01-16T20:25:30+0800 #7112 (Shi Yudi)
#    1. 适配1.x分支内部表格式
#    2. 去除0.5中的ConfigHelper，1.x只能显式指定ip等信息登录
#
# 2014-11-20    Shi Yudi
#    Last-Updated: 2014-11-20T15:26:41+0800 #6946 (Shi Yudi)
#    1. show only servers in current cluster in machine stat page
#    2. a little change for machine widget
#
# 2014-05-14    Shi Yudi
#    Last-Updated: 2014-05-14T16:15:26+0800 #6913 (Shi Yudi)
#    1. skip to select cluster if only one cluster
#
# 2014-04-24    Shi Yudi
#    Last-Updated: 2014-04-24T18:57:09+0800 #6515 (Shi Yudi)
#    1. support ob 0.5 updateserver columns
#
# 2014-04-22    Shi Yudi
#    Last-Updated: 2014-04-22T19:57:35+0800 #6288 (Shi Yudi)
#    1. add column filter for ups,ms,cs page
#
# 2014-04-01    Shi Yudi
#    Last-Updated: 2014-04-01T13:41:45+0800 #5844 (Shi Yudi)
#    1. fix oceanbase 0.5 gather stats mixed by all cluster
#
# 2014-02-27    Yudi Shi
#    Last-Updated: Thu Feb 27 20:01:13 2014 (+0800) #5777 (Yudi Shi)
#    1. add fail sql exec count in gallery page
#
# 2013-12-31    Shi Yudi
#    Last-Updated: Tue Dec 31 13:10:39 2013 (+0800) #5770 (Shi Yudi)
#    1. add error msg if geometry is not enough
#    2. fix some msg output
#
# 2013-12-11    Shi Yudi
#    Last-Updated: Wed Dec 11 14:51:43 2013 (+0800) #5745 (Shi Yudi)
#    1. fix password decrypt bug (fill 8 width encoding str)
#
# 2013-10-25    Shi Yudi
#    Last-Updated: Tue Dec  3 16:01:23 2013 (+0800) #5739 (Shi Yudi)
#    1. add time to widget frame
#
# 2013-10-25    Shi Yudi
#    Last-Updated: 2013-10-25 20:28:17 (+0800) #5647 (Shi Yudi)
#    1. add http api support
#    2. add obssh to login server
#    3. using obconfig password
#
# 2013-09-04    Shi Yudi
#    Last-Updated: 2013-09-04 10:52:02 (+0800) #4786 (Shi Yudi)
#    1. add machine stat
#    2. fix many bugs
#    3. add ssh, mysql login
#    4. add delete and restore widgets
#
# 2013-08-20    Shi Yudi
#    Last-Updated: 2013-08-20 15:29:45 (+0800) #2012 (Shi Yudi)
#    1. support instant statistics monitor for ups/cs/ms
#    2. keyboard response for switch between widget
#    3. rewrite help page
#
# 2013-08-16    Shi Yudi
#    Last-Updated: 2013-08-16 16:07:32 (+0800) #1052 (Shi Yudi)
#    1. add header column widget descripe each server
#    2. add instant stat list (only for ms now)
#    3. add mergeserver header info
#
# 2013-08-14    Shi Yudi
#    Last-Updated: 2013-08-14 16:11:40 (+0800) #576 (Shi Yudi)
#    1. add MessageBox for dooba, key 'p' for test
#    2. add selective mode, TAB for swtich between widgets
#    3. add supermode option, just a husk right now
#
# 2013-08-07    Shi Yudi
#    Last-Updated: 2013-08-07 10:38:33 (+0800) #479 (Shi Yudi)
#    1. fix term evironment setting bug
#    2. fix getting appname bug
#    3. refact main method
#
# 2013-08-06    Shi Yudi
#    Last-Updated: 2013-08-06 20:47:35 (+0800) #443 (Shi Yudi)
#    1. OceanBase alive checker before running
#    2. colorfull widgets
#    3. fix some promptions
#    4. change header widget and status widget style with horizontal line
#    5. remove page border
#
# 2013-08-06    Shi Yudi
#    Last-Updated: 2013-08-06 16:46:39 (+0800) #347 (Shi Yudi)
#    1. add chunkserver, mergeserver, updateserver info widgets
#    2. fix non-lexical closures problem
#    3. dynamic helper widget promption
#    4. change some pages' index
#
# 2013-07-31    Shi Yudi
#    Last-Updated: 2013-07-31 19:11:19 (+0800) #14 (Shi Yudi)
#    1. redesign helper bar and status bar
#    2. add more pages for dooba
#    3. beauty python coding style
#
# 2013-07-23    Shi Yudi
#    1. header with app name, and other mocks.
#    2. sql rt, sql count, cs rt, ups rt are added to screen.
#
#

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 3, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street, Fifth
# Floor, Boston, MA 02110-1301, USA.
#
#

# Code:

from collections import deque
from datetime import datetime, timedelta, date
from errno import *
from getopt import getopt, GetoptError
from locale import setlocale, LC_ALL
from os import environ, read, setsid
from pprint import pformat
from random import shuffle
from subprocess import Popen, PIPE, STDOUT, call
from telnetlib import Telnet
from threading import Thread
from time import sleep, strftime
from urllib2 import urlopen, URLError
import BaseHTTPServer
import atexit
import bz2
import curses
import curses.textpad
import itertools
import json
import math
import os
import select
import signal
import socket
import struct
import sys
import tempfile
import textwrap
import types
import traceback
import time

class Global:
    MAX_LINES = 100
    WIDGET_HEIGHT = 7
    DEFAULT_USER = 'root'
    DEFAULT_PASS = 'admin'


class Options(object):
    host = '127.0.0.1'
    port = 2828
    user = None
    password = ""
    database = "oceanbase"
    supermode = False
    interval = 1
    dataid = None
    using_ip_port = False
    env = 'unknown'
    machine_interval = 5
    degradation = False
    show_win_file = None
    daemon = False
    http = False
    daemon_action = 'start'
    # HTTP server relating
    http_port = 33244
    tenant_id = None
    tenant = None
    debug = False

    def __str__(self):
        result = {}
        for k in dir(self):
            v = getattr(self, k)
            if k.find('__') == 0:
                continue
            result[k] = v
        return pformat(result)

# auxiliary functions
class ColumnFactory(object):
    def __init__(self, svr, ip):
        self.__svr = svr
        self.__ip = ip

    def count(self, name, sname, obname, enable=False):
        DEBUG(self.count, "name,sname,obname", ",".join([str(name), str(sname), str(obname)]))
        if type(obname) == str:
            pass
        elif type(obname) == list:
            obname = [name for name in obname]
        else:
            raise Exception("unsupport type %s" % type(obname))
        DEBUG(self.count, "name,sname,obname", ",".join([str(name), str(sname), str(obname)]))
        return self.count0(name, sname, obname, enable)

    def count0(self, name, sname, obname, enable=False):
        DEBUG(self.count0, "name,sname,obname", ",".join([str(name), str(sname), str(obname)]))
        DEBUG(self.count0, "self.__svr|self.__ip", "|".join([self.__svr, self.__ip]))
        svr = self.__svr
        ip = self.__ip
        def calc_func(stat,ip=ip):
            def try_get(d, k, b):
                try:
                    if d.has_key(k):
                        return d[k]
                    else:
                        return d[b]
                except Exception as e:
                    raise e
            try:
                if type(obname) == str:
                    return try_get(stat, svr, oceanbase.get_current_tenant())[ip][obname]
                elif type(obname) == list:
                    return sum([try_get(stat, svr, oceanbase.get_current_tenant())[ip][name] for name in obname])
                else:
                    raise Exception("unsupport type %s" % type(obname))
            except KeyError as e:
#               DEBUG(calc_func, "exception", e)
                pass

        return Column(name, calc_func, 7, True, enable=enable, sname=sname)

    def time(self, name, sname, obnamet, obnamec=None, enable=False):
        if obnamec is None:
            obnamec = obnamet
        return self.time0(name, sname, obnamet, obnamec, enable)

    def time0(self, name, sname, obnamet, obnamec=None, enable=False):
        svr = self.__svr
        ip = self.__ip
        def calc_func(stat,ip=ip):
            def try_get(d, k, b):
                if d.has_key(k):
                    return d[k]
                else:
                    return d[b]
            try:
                if type(obnamec) == str:
                    total_count = try_get(stat, svr, oceanbase.get_current_tenant())[ip][obnamec]
                elif type(obnamec) == list:
                    total_count = sum([try_get(stat, svr, oceanbase.get_current_tenant())[ip][name] for name in obnamec])
                DEBUG(calc_func, "name", obnamet)
                DEBUG(calc_func, "ip", ip)
                DEBUG(calc_func, "svr", svr)
                DEBUG(calc_func, "enable", enable)
                DEBUG(calc_func, "tenant", oceanbase.get_current_tenant())
                DEBUG(calc_func, "try_get(stat, svr, oceanbase.get_current_tenant())[ip][obnamet]", try_get(stat, svr, oceanbase.get_current_tenant())[ip][obnamet])
                DEBUG(calc_func, "float(total_count)", float(total_count))
                DEBUG(calc_func, "try_get(stat, svr, oceanbase.get_current_tenant())[ip][obnamet] / 1000 / float(total_count)", try_get(stat, svr, oceanbase.get_current_tenant())[ip][obnamet] / 1000 / float(total_count or 1) )
                return try_get(stat, svr, oceanbase.get_current_tenant())[ip][obnamet] / 1000 / float(total_count or 1)
            except Exception as e:
#               DEBUG(calc_func, "exception", e)
                pass
        return Column(name, calc_func, 7, enable=enable, sname=sname)

    def cache(self, name, sname, obname):
        svr = self.__svr
        ip = self.__ip
        return Column(name, lambda stat,ip=ip: stat[svr][ip][obname+"_cache_hit"]
                      / float(stat[svr][ip][obname+"_cache_hit"] + stat[svr][ip][obname+"_cache_miss"] or 1),
                      7, enable=False, sname=sname)

def mem_str(mem_int, bit=False):
    mem_int = int(mem_int)
    if mem_int < 1024:
        return str(mem_int) + (bit and 'b' or '')
    mem_int = float(mem_int)
    mem_int /= 1024
    if mem_int < 1024:
        return "%.2fK" % mem_int + (bit and 'b' or '')
    mem_int /= 1024
    if mem_int < 1024:
        return "%.2fM" % mem_int + (bit and 'b' or '')
    mem_int /= 1024
    if mem_int < 1024:
        return "%.2fG" % mem_int + (bit and 'b' or '')
    return "UNKNOW"

def count_str(count_int, kilo=True):
    return str(count_int)

def percent_str(percent_int):
    return str(round(float(percent_int) * 100, 1)) + "%"

class Cowsay(object):
    '''Copyright 2011 Jesse Chan-Norris <jcn@pith.org>
       https://github.com/jcn/cowsay-py/blob/master/cowsay.py'''
    def __init__(self, str, length=40):
        self.__result = self.build_bubble(str, length) + self.build_cow()

    def __str__(self):
        return self.__result

    def build_cow(self):
        return """
         \   ^__^
          \  (oo)\_______
             (__)\       )\/\\
                 ||----w |
                 ||     ||
                """

    def build_bubble(self, str, length=40):
        bubble = []
        lines = self.normalize_text(str, length)
        bordersize = len(lines[0])
        bubble.append("  " + "_" * bordersize)

        for index, line in enumerate(lines):
            border = self.get_border(lines, index)
            bubble.append("%s %s %s" % (border[0], line, border[1]))
            bubble.append("  " + "-" * bordersize)

        return "\n".join(bubble)

    def normalize_text(self, str, length):
        lines  = textwrap.wrap(str, length)
        maxlen = len(max(lines, key=len))
        return [ line.ljust(maxlen) for line in lines ]

    def get_border(self, lines, index):
        if len(lines) < 2:
            return [ "<", ">" ]
        elif index == 0:
            return [ "/", "\\" ]
        elif index == len(lines) - 1:
            return [ "\\", "/" ]
        else:
            return [ "|", "|" ]


class Daemon:
        """
        A generic daemon class.

        Usage: subclass the Daemon class and override the run() method
        """
        def __init__(self, pidfile, stdin='/dev/null', stdout='/dev/null', stderr='/dev/null'):
                self.stdin = stdin
                self.stdout = stdout
                self.stderr = stderr
                self.pidfile = pidfile

        def daemonize(self):
                """
                do the UNIX double-fork magic, see Stevens' "Advanced
                Programming in the UNIX Environment" for details (ISBN 0201563177)
                http://www.erlenstar.demon.co.uk/unix/faq_2.html#SEC16
                """
                try:
                        pid = os.fork()
                        if pid > 0:
                                # exit first parent
                                sys.exit(0)
                except OSError, e:
                        sys.stderr.write("fork #1 failed: %d (%s)\n" % (e.errno, e.strerror))
                        sys.exit(1)

                # decouple from parent environment
                os.chdir("/")
                os.setsid()
                os.umask(0)

                # do second fork
                try:
                        pid = os.fork()
                        if pid > 0:
                                # exit from second parent
                                sys.exit(0)
                except OSError, e:
                        sys.stderr.write("fork #2 failed: %d (%s)\n" % (e.errno, e.strerror))
                        sys.exit(1)

                # redirect standard file descriptors
                sys.stdout.flush()
                sys.stderr.flush()
                si = file(self.stdin, 'r')
                so = file(self.stdout, 'a+')
                se = file(self.stderr, 'a+', 0)
                os.dup2(si.fileno(), sys.stdin.fileno())
                os.dup2(so.fileno(), sys.stdout.fileno())
                os.dup2(se.fileno(), sys.stderr.fileno())

                # write pidfile
                atexit.register(self.delpid)
                pid = str(os.getpid())
                file(self.pidfile,'w+').write("%s\n" % pid)

        def delpid(self):
                os.remove(self.pidfile)

        def start(self):
                """
                Start the daemon
                """
                # Check for a pidfile to see if the daemon already runs
                try:
                        pf = file(self.pidfile,'r')
                        pid = int(pf.read().strip())
                        pf.close()
                except IOError:
                        pid = None

                if pid:
                        message = "pidfile %s already exist. Daemon already running?\n"
                        sys.stderr.write(message % self.pidfile)
                        sys.exit(1)

                # Start the daemon
                self.daemonize()
                self.run()

        def stop(self):
                """
                Stop the daemon
                """
                # Get the pid from the pidfile
                try:
                        pf = file(self.pidfile,'r')
                        pid = int(pf.read().strip())
                        pf.close()
                except IOError:
                        pid = None

                if not pid:
                        message = "pidfile %s does not exist. Daemon not running?\n"
                        sys.stderr.write(message % self.pidfile)
                        return # not an error in a restart

                # Try killing the daemon process
                try:
                        while 1:
                                os.kill(pid, signal.SIGTERM)
                                sleep(0.1)
                except OSError, err:
                        err = str(err)
                        if err.find("No such process") > 0:
                                if os.path.exists(self.pidfile):
                                        os.remove(self.pidfile)
                        else:
                                print str(err)
                                sys.exit(1)

        def restart(self):
                """
                Restart the daemon
                """
                self.stop()
                self.start()

        def run(self):
                """
                You should override this method when you subclass Daemon. It will be called after the process has been
                daemonized by start() or restart().
                """

class OceanBase(object):
    instant_key_list = [
    # mergeserver
    'ms_memory_limit', 'ms_memory_total', 'ms_memory_parser',
    'ms_memory_transformer', 'ms_memory_ps_plan', 'ms_memory_rpc_request',
    'ms_memory_sql_array', 'ms_memory_expression', 'ms_memory_row_store',
    'ms_memory_session', 'ps_count',
    # chunkserver
    'serving_version', 'old_ver_tablets_num', 'old_ver_merged_tablets_num',
    'new_ver_tablets_num', 'new_ver_tablets_num', 'memory_used_default',
    'memory_used_network', 'memory_used_thread_buffer', 'memory_used_tablet',
    'memory_used_bi_cache', 'memory_used_block_cache',
    'memory_used_bi_cache_unserving', 'memory_used_block_cache_unserving',
    'memory_used_join_cache', 'memory_used_sstable_row_cache',
    'memory_used_merge_buffer', 'memory_used_merge_split_buffer',
    # updateserver
    'memory_total', 'memory_limit', 'memtable_total', 'memtable_used',
    'total_rows', 'active_memtable_limit', 'active_memtable_total',
    'active_memtable_used', 'active_total_rows', 'frozen_memtable_limit',
    'frozen_memtable_total', 'frozen_memtable_used', 'frozen_total_rows',
    'low_prio_queued_count', 'normal_prio_queued_count', 'high_prio_queued_count',
    'hotspot_queued_count',
    # machine stat
    'load1', 'load5', 'load15', 'MemTotal', 'MemFree',
    # mock
    'timestamp'
    ]

    app_info = {'username':None, 'password':None}

    def __init__(self, dataid=None):
        self.__q = {}
        self.__stop = True
        self.__machine_stat = {}
        self.__host = Options.host
        self.__port = Options.port
        self.update_dataid(dataid)
        self.__cur_cluster_svrs = None
        self.__cur_tenant_id = 1
        self.__cur_tenant_name = 'sys'
        self.tenant = []

    def update_dataid(self, dataid):
        if dataid:
            self.app_info = ObConfigHelper().get_app_info(dataid)

    def dosql(self, sql, host=None, port=None, database=None):
        if host is None:
            host = self.__host
        if port is None:
            port = self.__port
        if host is None:
            host = Options.host
        if port is None:
            port = Options.port
        if database is None:
            database = Options.database
        username = Options.user or self.app_info['username'] or Global.DEFAULT_USER
        password = Options.password
        if password:
            mysql = "mysql --connect_timeout=5 -s -N -h%s -P%d -u%s -p%s %s" % (host, port, username, password, database)
        else:
            mysql = "mysql --connect_timeout=5 -s -N -h%s -P%d -u%s %s" % (host, port, username, database)
        cmd = "%s -e \"%s\"" % (mysql, sql)
        DEBUG(self.dosql, "", cmd)
        p = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
        output = p.communicate()[0]
        err = p.wait()
        if err:
            raise Exception('popen Fail', cmd)
        return output

    def show_sql_result(self, sql, host=None, port=None, database=None):
        if host is None:
            host = self.__host
        if port is None:
            port = self.__port
        if host is None:
            host = Options.host
        if port is None:
            port = Options.port
        if database is None:
            database = Options.database
        username = Options.user or self.app_info['username'] or Global.DEFAULT_USER
        password = Options.password
        if password:
            mysql = "mysql --table --connect_timeout=5 --pager=less --column-names -h%s -P%d -u%s -p%s %s" % (host, port, username, password, database)
        else:
            mysql = "mysql --table --connect_timeout=5 --pager=less --column-names -h%s -P%d -u%s %s" % (host, port, username, database)
        cmd = "%s -e \"%s\" | less" % (mysql, sql)
        curses.endwin()
        call("clear")
        os.system(cmd)
        curses.doupdate()

    def mysql(self, host=None, port=None, database=None):
        if host is None:
            host = self.__host
        if port is None:
            port = self.__port
        if host is None:
            host = Options.host
        if port is None:
            port = Options.port
        if database is None:
            database = Options.database
        username = Options.user or self.app_info['username'] or Global.DEFAULT_USER
        password = Options.password
        if password:
            cmd = "mysql --connect_timeout=5 -h%s -P%d -u%s -p%s" % (host, port, username, password)
        else:
            cmd = "mysql --connect_timeout=5 -h%s -P%d -u%s" % (host, port, username)
        DEBUG(self.mysql, "mysql", cmd)
        environ['MYSQL_PS1'] = "(\u@\h) [%s]> " % self.app
        call('clear')
        try:
            call(cmd, shell=True)
        except KeyboardInterrupt:
            pass
        call('clear')

    def ssh(self, host):
        cmd = "ssh -o StrictHostKeyChecking=no %s" % host
        call('clear')
        try:
            call(cmd, shell=True)
        except KeyboardInterrupt:
            pass
        call('clear')

    def test_alive(self, fatal=True, do_false=None, do_true=None, host=None, port=None):
        if host is None:
            host = Options.host
        if port is None:
            port = Options.port
        try:
            oceanbase.dosql('select 1', host=host, port=port)
            if do_true is not None:
                do_true("Check oceanbase alive successfully! [%s:%d]" % (host, port))
        except Exception:
            if do_false is not None:
                do_false("Can't connect oceanbase, plz check options!\n"
                         "Options: [IP:%s] [PORT:%d] [USER:%s] [PASS:%s]"
                         % (host, port, Options.user, Options.password))
            if fatal:
                exit(1)
            else:
                return False
        return True

    def __check_schema(self):
        sql = """desc gv\\$sysstat"""
        res = self.dosql(sql)
        if "CON_ID" in res.split("\n")[0]:
            return """ select current_time(), con_id, svr_ip, svr_port, name, value from gv\$sysstat where con_id = %s"""
        else:
            return """ select current_time(), tenant_id, svr_ip, svr_port, stat_name, value from gv\$sysstat where tenant_id = %s"""

    def __get_all_stat(self):
        sql = self.__check_schema() % (str(self.get_current_tenant()))
        res = self.dosql(sql)
        r = dict()
        time = ''
        if not self.__using_server_time():
            time = str(datetime.now())
        for one in res.split("\n")[:-1]:
            a = one.split("\t")
            if self.__using_server_time():
                time = a[0]
            now = datetime.strptime(time, '%Y-%m-%d %H:%M:%S.%f')
            tnt = a[1]
            ip = a[2]
            port = a[3]
            name = a[4]
            value = int(a[5])
            ipport = ip + ":" + port
            a = [time, tnt, ip, port, name, value]
            DEBUG(self.__get_all_stat, "stuffs", a)
            if tnt not in r:
                r[tnt] = dict()
            if ipport not in r[tnt]:
                r[tnt][ipport] = dict()
            r[tnt][ipport][name] = value
            r['time'] = now
            r['timestamp'] = now
        return r

    def sub_stat(self, now, prev):
        r = self.__sub_stat(now, prev)
        return r

    def __sub_stat(self, now, prev):
        r = {}
        for k in now.keys():
            if k in self.instant_key_list:
                r[k] = now[k]
            elif type(now[k]) == dict:
                r.update({k: self.__sub_stat(now[k], prev[k])})
            elif type(now[k]) == int:
                r[k] = now[k] - prev[k]
            elif type(now[k]) == float:
                r[k] = now[k] - prev[k]
            elif type(now[k]) == type(datetime.min):
                r[k] = now[k] - prev[k]
            else:
                #print type(now[k])
                pass
        return r

    def __update_oceanbase_stat_runner(self):
        q = self.__q
        prev = self.__get_all_stat()
        while not self.__stop:
            sleep(Options.interval)
            if self.__stop:
                break
            try:
                cur = self.__get_all_stat()
            except Exception:
                continue
            if str(oceanbase.get_current_tenant()) not in cur:
                # tid has changed before result returned, ignore it.
                prev = self.__get_all_stat()
                continue
            try:
                q.append(self.__sub_stat(cur, prev))
                if (len(q) > Global.MAX_LINES):
                    q.popleft()
            except KeyError:
                pass
            prev = cur

    def __do_ssh(self, host, cmd):
        BUFFER_SIZE = 1 << 16
        ssh = Popen(['ssh', '-o', 'PreferredAuthentications=publickey',
                     '-o', 'StrictHostKeyChecking=no', host, cmd],
                    stdout=PIPE, stderr=PIPE, preexec_fn=setsid, shell=False)
        output = ssh.communicate()[0]
        err = ssh.wait()
        if err:
            raise Exception('popen Fail', "%s: %s" % (host, cmd))
        return output

    def __get_machine_stat(self, ip):
        cmd = ("cat <(date +%s%N) <(cat /proc/stat | head -1 | cut -d' ' -f 3-9) "
               "/proc/loadavg  <(cat /proc/meminfo) <(echo END_MEM) "
               "<(cat /proc/net/dev | sed -n 3~1p) <(echo END_NET) "
               "<(cat /proc/diskstats | grep '  8  ') <(echo END_DISK)")
        try:
            output = self.__do_ssh(ip, cmd)
        except Exception:
            return {}

        lines = output.split('\n')

        res = {}
        res['time'] = float(lines[0]) / 1000   # ms
        lines = lines[1:]

        cpuinfo = lines[0]
        columns = cpuinfo.split(" ")
        res['user'] = int(columns[0])
        res['nice'] = int(columns[1])
        res['sys'] = int(columns[2])
        res['idle'] = int(columns[3])
        res['iowait'] = int(columns[4])
        res['irq'] = int(columns[5])
        res['softirq'] = int(columns[6])
        res['total'] = 0
        for c in columns:
            res['total'] += int(c)
        lines = lines[1:]

        loadavg = lines[0]
        columns = loadavg.split(" ")
        res['load1'] = columns[0]
        res['load5'] = columns[1]
        res['load15'] = columns[2]
        lines = lines[1:]

        # mem
        for line in lines:
            if 'END_MEM' == line:
                break
            kv = line.split()
            k = kv[0][:-1]
            v = kv[1]
            res[k] = int(v) * 1024
        idx = lines.index('END_MEM')
        lines = lines[idx + 1:]

        # net
        res['net'] = {}
        for line in lines:
            if 'END_NET' == line:
                break
            colon = line.find(':')
            assert colon > 0, line
            name = line[:colon].strip()
            res['net'][name] = {}
            fields = line[colon+1:].strip().split()
            res['net'][name]['bytes_recv'] = int(fields[0])
            res['net'][name]['packets_recv'] = int(fields[1])
            res['net'][name]['errin'] = int(fields[2])
            res['net'][name]['dropin'] = int(fields[3])
            res['net'][name]['bytes_sent'] = int(fields[8])
            res['net'][name]['packets_sent'] = int(fields[9])
            res['net'][name]['errout'] = int(fields[10])
            res['net'][name]['dropout'] = int(fields[11])
        idx = lines.index('END_NET')
        lines = lines[idx + 1:]

        res['disk'] = {}
        SECTOR_SIZE = 512
        for line in lines:
            # http://www.mjmwired.net/kernel/Documentation/iostats.txt
            if 'END_DISK' == line:
                break
            _, _, name, reads, _, rbytes, rtime, writes, _, wbytes, wtime = line.split()[:11]
            res['disk'][name] = {}
            res['disk'][name]['rbytes'] = int(rbytes) * SECTOR_SIZE
            res['disk'][name]['wbytes'] = int(wbytes) * SECTOR_SIZE
            res['disk'][name]['reads'] = int(reads)
            res['disk'][name]['writes'] = int(writes)
            res['disk'][name]['rtime'] = int(rtime)
            res['disk'][name]['wtime'] = int(wtime)
        return res
        idx = lines.index('END_DISK')
        lines = lines[idx + 1:]

    def __update_server_stat_runner(self):
        prev = {}
        for ip in self.ip_list:
            prev[ip] = self.__get_machine_stat(ip)
        while not self.__stop:
            sleep(Options.machine_interval)
            shuffle(self.ip_list)
            for ip in self.ip_list:
                if self.__stop:
                    break
                cur = self.__get_machine_stat(ip)
                try:
                    result = self.__sub_stat(cur, prev[ip])
                    self.__machine_stat.update({ip: result})
                except KeyError:
                    pass
                prev[ip] = cur

    def __update_version(self):
        self.version = "Unknown"
        res = self.dosql("show variables like 'version_comment'")
        for one in res.split("\n")[:-1]:
            a = one.split("\t")
            self.version = a[1]
        self.app = oceanbase.dosql("select value from __all_virtual_sys_parameter_stat where name='cluster' limit 1")[:-1]

    def __using_server_time(self):
        if self.version.find('0.4.1') >= 0:
            return False
        elif self.version.find('0.4.2') >= 0:
            return True
        return False

    def start(self):
        self.__stop = False
        self.__q = deque([])
        self.__th = Thread(target=self.__update_oceanbase_stat_runner, args=())
        self.__th.daemon = True
        self.__th.start()
        if Options.env == 'online':
            self.__update_ip_list()
            self.__machine_stat = {}
            self.__svr_th = Thread(target=self.__update_server_stat_runner, args=())
            self.__svr_th.daemon = True
            self.__svr_th.start()

    def dump_queue(self):
        print self.__q

    def now(self):
        return self.__q[-1]

    def stat_count(self):
        return len(self.__q)

    def latest(self, num=1):
        return list(self.__q)[-num:]

    def machine_stat(self):
        return self.__machine_stat

    def update_tenant_info(self):
        DEBUG(self.update_tenant_info, "update_tenant_info", "start update_tenant_info")
        class Tenant:
            def __init__(self, tid, name, zone_list, selected):
                self.tenant_id = tid
                self.tenant_name = name
                self.zone_list = zone_list
                self.selected = selected
                self.svr_list = {"observer":[]}
        res = self.dosql("select tenant_id, tenant_name, zone_list from __all_tenant")
        self.tenant = []
        flg = True
        for line in res.split("\n")[0:-1]:
            tnt = line.split("\t")
            self.tenant.append(Tenant(tnt[0], tnt[1], tnt[2], flg))
            flg = False
        DEBUG(self.update_tenant_info, "self.tenant", self.tenant)

        svrs = self.dosql("select svr_ip, svr_port, id, zone, inner_port, with_rootserver, status from __all_server")
        for line in svrs.rstrip("\n").split("\n"):
            svr = line.split("\t")
            for tnt in self.tenant:
                tnt.svr_list["observer"].append({"ip":svr[0], "port":svr[1], "role":svr[5]})

        self.__update_version()
        #self.__update_ip_list()
        self.__update_cur_cluster_info()
        self.__update_ip_list()
        self.__update_sample()

    def check_lms(self, say):
        if Options.dataid is not None:
            lms_list = self.app_info['lms_list']
            if not lms_list or len(lms_list) <= 0:
                say('Get lms list fail, plz check'
                    + ' [ dataid = %s, lms_list = %s ]' % (Options.dataid, lms_list))
            else:
                for lms in lms_list:
                    say('checking lms [%s:%s]' % lms)
                    if oceanbase.test_alive(host=lms[0], port=lms[1], fatal=False,
                                            do_false=say, do_true=say):
                        self.__host = lms[0]
                        self.__port = lms[1]
                        #oceanbase.update_cluster_info()
                        oceanbase.update_tenant_info()
                        return True
        return False

    def __update_sample(self):
        self.sample = self.__get_all_stat()
        self.sample['time'] = timedelta()

    def __update_ip_list(self):
        ip_list = []
        ip_map = {}
        for clu in self.tenant:
            svr_list = clu.svr_list
            for name in ["observer"]:
                ip_list += [svr["ip"] for svr in svr_list[name]]
                for ip in [svr["ip"] for svr in svr_list[name]]:
                    if ip in ip_map:
                        ip_map[ip].append(name)
                    else:
                        ip_map[ip] = [name]
        self.ip_list = list(set(ip_list))
        self.ip_map = ip_map

#   def __update_ip_list(self):
#       ip_list = []
#       ip_map = {}
#       for clu in self.cluster:
#           svr_list = clu.svr_list
#           for name in ('chunkserver', 'mergeserver', 'updateserver', 'rootserver'):
#               ip_list += [svr['ip'] for svr in svr_list[name]]
#               for ip in [svr['ip'] for svr in svr_list[name]]:
#                   if ip in ip_map:
#                       ip_map[ip].append(name)
#                   else:
#                       ip_map[ip] = [name]
#       self.ip_list = list(set(ip_list))
#       self.ip_map = ip_map

    def __update_cur_cluster_info(self):
        for tnt in self.tenant:
            self.__cur_tenant_id = tnt.tenant_id
            self.__cur_cluster_svrs = tnt.svr_list
            return self.__cur_cluster_svrs

#   def __update_cur_cluster_info(self):
#       ip = self.__host
#       port = self.__port
#       for clu in self.cluster:
#           for ms in clu.svr_list['mergeserver']:
#               if ms['ip'] == ip and ms['port'] == port:
#                   if Options.dataid:
#                       self.__host, self.__port = clu.vip, clu.port
#                   self.__cur_cluster_svrs = clu.svr_list
#                   self.__cur_cluster_id = clu.id
#                   return self.__cur_cluster_svrs
#           if clu.vip == ip and clu.port == port:
#               if Options.dataid:
#                   self.__host, self.__port = clu.vip, clu.port
#               self.__cur_cluster_svrs = clu.svr_list
#               self.__cur_cluster_id = clu.id
#               return self.__cur_cluster_svrs
#       return {}

    def update_lms(self):
        svrs = oceanbase.dosql("select svr_ip, svr_port from __all_server where with_rootserver = 1 limit 1")
        for line in svrs.rstrip("\n").split("\n"):
            svr = line.split("\t")
            ip = svr[0]
            try:
                port = int(svr[1])
                self.__host = svr[0]
                self.__port = svr[1]
            except ValueError:
                return

#   def update_lms(self):
#       svrs = oceanbase.dosql("select svr_ip,svr_port from (select * from __all_server_stat) t where svr_type='updateserver' limit 1")
#       for line in svrs.rstrip("\n").split("\n"):
#           svr = line.split("\t")
#           ip = svr[0]
#           try:
#               port = int(svr[1])
#           except ValueError:            # no lms
#               return
#       for clu in self.cluster:
#           for ups in clu.svr_list['updateserver']:
#               if ups['ip'] == ip and ups['port'] == port:
#                   self.__host = clu.vip
#                   self.__port = clu.port
#                   return

    def __set_current_tenant(self, tname, tid):
        self.__cur_tenant_name = tname
        self.__cur_tenant_id = tid

    def get_current_tenant(self):
        return self.__cur_tenant_id

    def get_current_tenant_name(self):
        return self.__cur_tenant_name

    def get_tenant_svr(self, tid=None):
        for tnt in self.tenant:
            DEBUG(self.get_tenant_svr, "cur_tenant_id.selected", str(tnt.tenant_id) + "|" + str(tnt.selected))
            if True == tnt.selected:
                return tnt.svr_list

    def find_svr_list(self):
        if not self.__cur_cluster_svrs:
            self.__update_cur_cluster_info()
        return self.__cur_cluster_svrs

    def stop(self):
        self.__stop = True

    def switch_tenant(self, tid):
        tname = "sys"
        for tnt in self.tenant:
            if str(tid) == str(tnt.tenant_id) or tid == tnt.tenant_name:
                tname = tnt.tenant_name
                tid = tnt.tenant_id
                tnt.selected = True
                self.__q.clear()
            else:
                tnt.selected = False
        DEBUG(self.switch_tenant, "self.tenant.selected", [" ".join([tnt.tenant_id, str(tnt.selected)]) for tnt in self.tenant])
        self.__update_cur_cluster_info()
        self.__update_sample()
        self.__set_current_tenant(tname, tid)


class Page(object):
    def __init__(self, parent, layout, y, x, height, width):
        self.__parent = parent
        self.__layout = layout
        self.__widgets = []
        self.__win = parent.derwin(height, width, y, x)
        self.__y = y
        self.__x = x
        self.__height = height
        self.__width = width
        self.border()
        self.__win.nodelay(1)
        self.__win.timeout(0)
        self.__win.keypad(1)
        self.move(y, x)
        self.resize(height, width)
        self.__cur_select = 0
        self.__shown_widget_num = 0

    def add_widget(self, widget):
        if 0 == len(self.__widgets):
            widget.select(True)
        self.__widgets.append(widget)
        self.__rearrange()

    def update_widgets(self):
        pass

    def __rearrange(self):
        undeleted_widgets = filter(lambda w: False == w.delete(), self.__widgets)
        self.__layout.rearrange(0, 0, self.__height, self.__width, undeleted_widgets)

    def rearrange(self):
        self.__rearrange()

    def update(self):
        self.update_widgets()

    def clear_widgets(self):
        self.__widgets = []

    def resize(self, height, width):
        self.__height = height
        self.__width = width
        self.__win.resize(height, width)
        self.__rearrange()

    def move(self, y, x):
        self.__x = x
        self.__y = y
        self.__win.mvderwin(y, x)
        self.__rearrange()

    def __reset_widgets(self):
        if len(self.__widgets) <= 0:
            return
        shown_widgets = self.shown_widgets()
        map(lambda w: w.select(False), self.__widgets)
        map(lambda w: w.delete(False), self.__widgets)
        self.__cur_select = 0
        self.__widgets[self.__cur_select].select(True)
        self.__rearrange()

    def __delete_current_widget(self):
        shown_widgets = self.shown_widgets()
        if len(shown_widgets) <= 0:
            return
        shown_widgets[self.__cur_select].select(False)
        shown_widgets[self.__cur_select].delete(True)
        self.__cur_select -= 1
        self.select_next()
        self.__rearrange()

    def border(self):
        pass

    def process_key(self, ch):
        if ch == ord('d'):
            self.__delete_current_widget()
        elif ch == ord('R'):
            self.__reset_widgets()
        elif ch == ord('m'):
            curses.endwin()
            oceanbase.mysql()
            curses.doupdate()
        elif ch == ord('j'):
            curses.endwin()
            w = self.selected_widget()
            oceanbase.ssh(w.host())
            curses.doupdate()

    def redraw(self):
        self.erase()
        self.__layout.redraw(self.shown_widgets())

    def getch(self):
        return self.__win.getch()

    def erase(self):
        self.__win.erase()

    def win(self):
        return self.__win

    def title(self):
        return 'Untitled'

    def select_next(self):
        shown_widgets = self.shown_widgets()
        if len(shown_widgets) <= 0:
            return
        shown_widgets[self.__cur_select].select(False)
        self.__cur_select = (self.__cur_select + 1) % len(shown_widgets)
        shown_widgets[self.__cur_select].select(True)

    def parent(self):
        return self.__parent

    def shown_widgets(self):
        '''Actually shown widgets that is all widgets except for
        1. deleted widgets and,
        2. couldn\'t display widgets as no space for them.
        '''
        return filter(lambda w: w.show() and False == w.delete(), self.__widgets)

    def valid_widgets(self):
        '''All widgets excpet for the deleted widgets.'''
        return filter(lambda w: False == w.delete(), self.__widgets)

    def all_widgets(self):
        return self.__widgets

    def selected_widget(self):
        '''
        may has no selected widget
        '''
        if len(self.shown_widgets()) > self.__cur_select:
            return self.shown_widgets()[self.__cur_select]
        return None

    def select_columns(self):
        if len(self.all_widgets()) > 0:
            all_widgets = [hc_widget.column_widget() for hc_widget in self.all_widgets()]
            columns = all_widgets[0].valid_columns()
            columns_ret = ColumnCheckBox('Select Columns', columns, self.__parent).run()
            for widget in all_widgets:
                for idx in range(0, len(columns)):
                    widget.valid_columns()[idx].enable(columns[idx].enable(enable=True))
                widget.update()
            [hc_widget.resize() for hc_widget in self.all_widgets()]
            self.rearrange()


class Layout(object):
    def __init__(self):
        pass

    def redraw(self, widgets):
        for widget in widgets:
            try:
                if widget.show():
                    widget.redraw()
            except Exception:
                pass

    def __calc_widget_height(self, height, width, widgets):
        max_min_height = max([ widget.min_height() for widget in widgets ])
        for widget in widgets:
            widget.min_height(max_min_height)

        wwidths = [ widget.min_width() for widget in widgets ]
        cur_width = 0
        nline = 1
        for wwidth in wwidths:
            if cur_width + wwidth <= width:
                cur_width += wwidth
            else:
                cur_width = wwidth
                nline += 1
        widget_height = height / nline
        return max(widget_height, max_min_height)

    def rearrange(self, y, x, height, width, widgets):
        if height <= 0 or width <= 0 or len(widgets) <= 0:
            return 0
        widget_height = self.__calc_widget_height(height, width, widgets)

        for widget in widgets:
            widget.show(False)
        cur_y = 0
        cur_x = 0
        for index,widget in enumerate(widgets):
            if cur_x + widget.min_width() > width:
                cur_y += widget_height
                cur_x = 0
            try:
                widget.move(0, 0)
                widget.resize(widget_height, widget.min_width())
                widget.move(cur_y + y, cur_x + x)
                widget.show(True)
            except curses.error:
                return index
            cur_x += widget.min_width()
        return len(widgets)


class Widget(object):
    def __init__(self, min_height, min_width, parent, use_win=False):
        if use_win:
            self.__win = parent
        else:
            self.__win = parent.derwin(min_height, min_width, 0, 0)
        self.__min_height = min_height
        self.__min_width = min_width
        self.__height, self.__width = self.__win.getmaxyx()
        self.__y, self.__x = self.__win.getmaxyx()
        self.__select = False
        self.__show = False
        self.__deleted = False

    def resize(self, height, width):
        self.__height = height
        self.__width = width
        maxh, maxw = self.__win.getmaxyx()
        if (width > maxw):
            MessageBox(self.__win, "TERM is too small!").run(anykey=True)
            exit(1)
        self.__win.resize(height, width)

    def move(self, y, x):
        self.__win.mvderwin(y, x)
        self.__y = y
        self.__x = x

    def mvwin(self, y, x):
        self.__win.mvwin(y, x)
        self.__y = y
        self.__x = x

    def min_height(self, height=None):
        if height:
            self.__min_height = height
        return self.__min_height

    def min_width(self, width=None):
        if width:
            self.__min_width = width
        return self.__min_width

    def geometry(self):
        return self.__y, self.__x, self.__height, self.__width

    def height(self):
        return self.__height

    def width(self):
        return self.__width

    def redraw(self):
        pass

    def refresh(self):
        self.__win.refresh()

    def erase(self):
        self.__win.erase()

    def win(self):
        return self.__win

    def select(self, select = None):
        if select is not None:
            self.__select = select
        return self.__select

    def win(self):
        return self.__win

    def show(self, show=None):
        if show is not None:
            self.__show = show
        return self.__show

    def update(self):
        pass

    def delete(self, delete=None):
        if delete is not None:
            self.__deleted = delete
        return self.__deleted


class Column(object):
    def __init__(self, name, filter, width, duration=False, enable=True, sname=None):
        self.__name = name
        self.__filter = filter
        self.__width = width
        self.__duration = duration
        self.__enable = enable
        self.__sname = sname or name
        self.__valid = True
        DEBUG(self.__init__, "self.__name, self.__sname", ",".join([self.__name, self.__sname]))

    def __str__(self):
        return self.name() + " (" + self.sname() + ")"

    def __repr__(self):
        return self.name()

    def __eq__(self, obj):
        return isinstance(obj, Column) and self.name() == obj.name()

    def name(self):
        return self.__name

    def sname(self):
        return self.__sname

    def header(self):
        return self.__sname.center(self.__width).upper()

    def value(self, stat):
        def seconds(td):
            return float(td.microseconds +
                         (td.seconds + td.days * 24 * 3600) * 10**6) / 10**6;

        d = self.__filter(stat)
        if type(d) == type(0.0):
            div = (seconds(stat["timestamp"]) or 1.0) if self.__duration else 1.0
            v = d / div
            if (v > 100):
                v = int(v)
                return str(v).center(self.__width)[:self.__width]
            return ("%.2f" % v).center(self.__width)[:self.__width]
        elif type(d) == type(0):
            div = (seconds(stat['time']) or 1.0) if self.__duration else 1.0
            v = int(d / div)
            if (self.__sname == 'ni' or self.__sname == 'no'):
                return mem_str(v).center(self.__width)[:self.__width]
            return str(v).center(self.__width)[:self.__width]
        elif type(d) == type(''):
            return d.center(self.__width)[:self.__width]

    def enable(self, enable=None):
        if enable is not None:
            self.__enable = enable
        return self.__enable

    def valid(self, valid=None):
        if valid is not None:
            self.__valid = valid
        return self.__valid

class MachineStatWidget(Widget):
    def __init__(self, name, parent, border=True):
        self.__name = name
        self.__border = border
        width = 48
        Widget.__init__(self, 20, width, parent)

    def host(self):
        return self.__name.split(':')[0]

    def type_str(self):
        return self.__name.split(':')[1]

    def redraw(self):
        if self.__border:
            if self.select():
                self.win().attron(curses.color_pair(1) | curses.A_BOLD)
                self.win().box()
                self.win().attroff(curses.color_pair(1) | curses.A_BOLD)
            else:
                self.win().box()
            self.win().addstr(0, 2, " " + self.__name + " ", curses.color_pair(3))

        try:
            stat = oceanbase.machine_stat()[self.host()]
        except KeyError:
            return

        if self.__border:
            # cpu info
            self.win().addstr(1, 2, "%-7s%4s%%" % ("CPU", "100"), curses.color_pair(7) | curses.A_BOLD)
            # self.win().attron(curses.color_pair(8))
            self.win().addstr(1, 17, "%-7s%4.1f%%" % ("nice:", float(stat['nice']) / stat['total'] * 100))
            self.win().addstr(2, 2, "%-7s%4.1f%%" % ("user:", float(stat['user']) / stat['total'] * 100))
            self.win().addstr(2, 17, "%-7s%4.1f%%" % ("iowait:", float(stat['iowait']) / stat['total'] * 100))
            self.win().addstr(3, 2, "%-7s%4.1f%%" % ("sys:", float(stat['sys']) / stat['total'] * 100))
            self.win().addstr(3, 17, "%-7s%4.1f%%" % ("irq:", float(stat['irq']) / stat['total'] * 100))
            self.win().addstr(4, 2, "%-7s%4.1f%%" % ("idle:", float(stat['idle']) / stat['total'] * 100))
            self.win().addstr(4, 17, "%-7s%4.1f%%" % ("sirq:", float(stat['softirq']) / stat['total'] * 100))
            # self.win().attroff(curses.color_pair(8))

            # load
            self.win().addstr(1, 32, "%-7s%6s" % ("Load", "x-core"), curses.color_pair(7) | curses.A_BOLD)
            # self.win().attron(curses.color_pair(8))
            self.win().addstr(2, 32, "%-7s%6s" % ("1 min:", stat['load1']))
            self.win().addstr(3, 32, "%-7s%6s" % ("5 min:", stat['load5']))
            self.win().addstr(4, 32, "%-7s%6s" % ("15 min:", stat['load15']))
            # self.win().attroff(curses.color_pair(8))

            # mem info
            self.win().addstr(6, 2, "%-6s%7s" % ("Mem", mem_str(stat['MemTotal'])), curses.color_pair(7) | curses.A_BOLD)
            self.win().addstr(7, 2, "%-6s%7s" % ("used:", mem_str(stat['MemTotal'] - stat['MemFree'])))
            self.win().addstr(8, 2, "%-6s%7s" % ("free:", mem_str(stat['MemFree'])))

            # net info
            self.win().addstr(6, 17, "%-9s%9s%9s" % ("Network", 'Rx/s', 'Tx/s'), curses.color_pair(7) | curses.A_BOLD)
            for idx,item in enumerate(stat['net'].items()):
                if idx > 5:
                    break
                self.win().addstr(7 + idx, 17, "%-9s%9s%9s" %
                                  (item[0],
                                   mem_str(item[1]['bytes_recv'] * pow(10, 6) / float(stat['time']), bit=True),
                                   mem_str(item[1]['bytes_sent'] * pow(10, 6) / float(stat['time']), bit=True)))

            # disk io info
            self.win().addstr(14, 2, "%-5s%6s%9s" % ("Disk I/O", 'In/s', 'Out/s'), curses.color_pair(7) | curses.A_BOLD)
            idx = 15
            for item in stat['disk'].items():
                if idx >= self.height() - 1:
                    break
                if item[1]['rbytes'] <= 0 and item[1]['wbytes'] <= 0:
                    continue
                self.win().addstr(idx, 2, "%-5s%9s%9s" %
                                  (item[0],
                                   mem_str(item[1]['rbytes'] * pow(10, 6) / float(stat['time'])),
                                   mem_str(item[1]['wbytes'] * pow(10, 6) / float(stat['time']))))
                idx += 1

row_data = {}
class ColumnWidget(Widget):
    def __init__(self, name, columns, parent, border=True):
        self.__check_column_valid(columns)
        self.__name = name
        self.__columns = columns
        self.__border = border
        enabled_columns = filter(lambda c: c.enable(), self.valid_columns())
        width = len(" ".join([c.header() for c in enabled_columns])) + 2 # 2 padding
        width = max(width, 18)
        if border:
            width += 2
        Widget.__init__(self, Global.WIDGET_HEIGHT, width, parent)

    def __check_column_valid(self, columns):
        for c in columns:
            try:
                c.value(oceanbase.sample)
            except KeyError:
                c.valid(False)

    def valid_columns(self):
        return filter(lambda c: c.valid(), self.__columns)

    def update(self):
        enabled_columns = filter(lambda c: c.enable(), self.valid_columns())
        width = len(" ".join([c.header() for c in enabled_columns])) + 2 # 2 border + 2 padding
        if self.__border:
            width += 2
        self.min_width(width)

    def redraw(self):
        def stat_count():
            return oceanbase.stat_count()
        def latest(num):
            return oceanbase.latest(num)
        enabled_columns = filter(lambda c: c.enable(), self.valid_columns())
        if self.__border:
            if self.select():
                self.win().attron(curses.color_pair(1) | curses.A_BOLD)
                self.win().box()
                self.win().attroff(curses.color_pair(1) | curses.A_BOLD)
            else:
                self.win().box()
            if self.__name:
                self.win().addstr(0, 2, " " + self.__name + " ", curses.color_pair(3))
            print_lines = min(stat_count(), self.height() - 3)
            self.win().addstr(1, 1, " " + " ".join([ c.header() for c in enabled_columns ]) + " ", curses.color_pair(2))
            border_offset = 1
        else:
            print_lines = min(stat_count(), self.height() - 1)
            self.win().addstr(0, 0, " " + " ".join([ c.header() for c in enabled_columns ]) + " ", curses.color_pair(2))
            border_offset = 0
        latest = latest(print_lines)
        li = []
        # DEB(enabled_columns,enabled_columns)
        for i in range(0, len(latest)):
            item = []
            for c in enabled_columns:
                item.append(str(c.value(latest[i])))
            li.append(" " + " ".join(item))
            for i in range(0, len(li)):
                self.win().addstr(i + 1 + border_offset, border_offset, li[i])
        
        # DEB(item,item)


class HeaderColumnWidget(Widget):
    def __init__(self, name, columns, parent, padding=0, get_header=None):
        self.__name = name
        self.__padding = padding
        self.__get_header = get_header
        Widget.__init__(self, Global.WIDGET_HEIGHT, 0, parent)
        self.__column_widget = ColumnWidget(None, columns, self.win(), border=False)
        self.min_width(self.__column_widget.min_width() + 2)
        self.resize(self.min_height(), self.min_width())

    def redraw(self):
        if self.select():
            self.win().attron(curses.color_pair(1))
            self.win().box()
            self.win().attroff(curses.color_pair(1))
        else:
            self.win().box()

        nline = 1
        if self.__get_header and oceanbase.stat_count() > 0:
            header = self.__get_header(oceanbase.now())
            string = ""
            for item in header.items():
                item_str = ": ".join([item[0], str(item[1])])
                if (len(string) + len(item_str) + 2 > self.min_width() - 2):
                    string = string[2:]
                    self.win().addstr(nline, 1, string.center(self.min_width() - 2))
                    string = ""
                    nline += 1
                string += "  " + item_str
            if string:
                string = string[2:]
                self.win().addstr(nline, 1, string.center(self.min_width() - 2))
                self.__set_padding(nline)
            else:                         # fix ob 0.4.1 wired bug
                self.__set_padding(0)
        self.__column_widget.redraw()
        if self.__name:
            self.win().addstr(0, 2, " " + self.__name + " ", curses.color_pair(3))
        time = strftime(" %Y-%m-%d %T ")
        self.win().addstr(0, self.width() - len(time) - 2, time, curses.color_pair(3))

    def resize(self, height=None, width=None):
        if height is not None and width is not None:
            Widget.resize(self, height, width)
            maxy, maxx = self.win().getmaxyx()
            self.__column_widget.resize(maxy - self.__padding - 2, maxx - 2)
        if height is None:
            height = self.__column_widget.min_height() + self.__padding + 2
            self.resize(height, width)
        if width is None:
            width = self.__column_widget.min_width() + 2
            self.min_width(width)
            self.resize(height, width)

    def move(self, y, x):
        self.win().mvderwin(y, x)
        self.__column_widget.win().mvderwin(self.__padding + 1, 1)

    def __set_padding(self, padding):
        self.min_height(padding + 2 + 2)
        maxy, maxx = self.win().getmaxyx()
        self.__padding = padding
        self.__column_widget.move(0, 0)
        self.__column_widget.resize(maxy - self.__padding - 2, maxx - 2)
        self.__column_widget.move(self.__padding + 1, 1)

    def update(self):
        self.__column_widget.update()

    def host(self):
        return self.__name.split(":")[0]

    def sql_port(self):
        return self.__name.split(":")[1]

    def column_widget(self):
        return self.__column_widget

class StatusWidget(Widget):
    def __init__(self, parent):
        self.__parent = parent
        maxy, maxx = parent.getmaxyx()
        Widget.__init__(self, 2, maxx, parent)

        self.resize(2, maxx)
        self.move(maxy - 2, 0)
        self.win().bkgd(curses.color_pair(1))

    def redraw(self):
        now = strftime("%Y-%m-%d %T")
        maxy, maxx = self.win().getmaxyx()
        tps = self.__get_tps()
        qps = self.__get_qps()
        svr_list = oceanbase.find_svr_list()["observer"][0]
        statstr = "HOST: %s:%d " % (Options.host, Options.port)
        statstr += "%s<%s> TPS: %-6d QPS: %-6d " % (oceanbase.app, oceanbase.get_current_tenant_name(), tps, qps)
        statstr += "SERVERS: %d" % (len(svr_list))
        try:
            self.win().addstr(1, maxx - len(now) - 2, now)
            self.win().addstr(1, 2, statstr)
            self.win().hline(0, 0, curses.ACS_HLINE, 1024)
        except curses.error:
            pass

    def __seconds(self, td):
        return float(td.microseconds +
                     (td.seconds + td.days * 24 * 3600) * 10**6) / 10**6;

    def __get_tps(self):
        tid = str(oceanbase.get_current_tenant())
        if oceanbase.stat_count() > 1:
            last = oceanbase.now()
            insert = sum([item["sql insert count"]
                            for item in last[tid].values()])
            replace = sum([item["sql replace count"]
                            for item in last[tid].values()])
            delete = sum([item["sql delete count"]
                            for item in last[tid].values()])
            update = sum([item["sql update count"]
                            for item in last[tid].values()])
            tps = float(insert + replace + delete + update) / self.__seconds(last['time'])
            return int(tps)
        else:
            return 0

    def __get_qps(self):
        tid = str(oceanbase.get_current_tenant())
        if oceanbase.stat_count() > 1:
            last = oceanbase.now()
            total = float(sum([item["sql select count"]
                               for item in last[tid].values()]))
            qps = total / self.__seconds(last['time'])
            return int(qps)
        else:
            return 0


class HeaderWidget(Widget):
    def __init__(self, parent):
        maxy, maxx = parent.getmaxyx()
        Widget.__init__(self, 2, maxx, parent)
        self.__page = None
        self.win().bkgd(curses.color_pair(1))

    def redraw(self):
        self.erase()
        if self.__page is None:
            try:
                self.win().addstr(0, 2, '1:Help 2:Gallery 3:SQL(ObServer)', curses.A_BOLD)
                self.win().hline(1, 0, curses.ACS_HLINE, 1024, curses.A_BOLD)
            except curses.error:
                pass
        else:
            maxy, maxx = self.win().getmaxyx()
            shown_num = len(self.__page.shown_widgets())
            valid_num = len(self.__page.valid_widgets())
            all_num = len(self.__page.all_widgets())
            shown_str = 'Shown: %d / Valid: %d / Total: %d' % (shown_num, valid_num, all_num)
            try:
                self.win().addstr(0, 2, self.__page.title(), curses.A_BOLD)
                self.win().addstr(0, maxx - len(shown_str) - 4, shown_str, curses.A_BOLD)
                self.win().hline(1, 0, curses.ACS_HLINE, 1024, curses.A_BOLD)
            except curses.error:
                pass

    def switch_page(self, page):
        self.__page = page


class MessageBox(object):
    def __init__(self, parent, msg):
        self.msg = msg
        self.parent = parent
        maxy, maxx = parent.getmaxyx()
        width = min(maxx - 20, len(msg)) + 10
        height = 5
        y = (maxy - height) / 2
        x = (maxx - width) / 2
        self.win = parent.derwin(height, width, y, x)
        self.win.erase()
        self.win.attron(curses.color_pair(1))
        self.win.box()
        self.win.attroff(curses.color_pair(1))
        self.win.addstr(2, 4, self.msg, curses.color_pair(0))

    def run(self, anykey=False):
        res = True
        while (1):
            c = self.win.getch()
            if anykey:
                break
            elif c == ord('y'):
                res = True
                break
            elif c == ord('n'):
                res = False
                break
            elif c == ord('q'):
                res = False
                break
        self.win.erase()
        self.win.refresh()
        return res

class PopPad(Widget):
    def __init__(self, name, items, parent, **args):
        self.__offset_x = 0
        self.__offset_y = 0
        self.__name = name

        maxy, maxx = parent.getmaxyx()
        width = maxx - 100
        height = 10
        Widget.__init__(self, height, width, parent)
        self.resize(height, width)
        y = (maxy - height - self.__offset_y) / 2
        x = (maxx - width - self.__offset_x) / 2
        self.mvwin(self.__offset_y + y, self.__offset_x + x)
        self.move(self.__offset_y + y, self.__offset_x + x)

    def __do_key(self):
        ch = self.win().getch()
        stopFlg = False
        if ch in (ord("q"), ord("Q")):
            stopFlg = True

    def __redraw(self):
        self.win().erase()
        self.win().attron(curses.color_pair(1))
        self.win().box()
        self.win().attroff(curses.color_pair(1))
        self.win().addstr(0, 2, ' %s ' % self.__name, curses.color_pair(3))

    def run(self):
        self.__redraw()
        while(True):
            if self.__do_key():
                self.win().erase()
                self.win().refresh()
                return
            self.__redraw()
            self.win().refresh()

class SelectionBox(Widget):
    __padding_left = 5
    __padding_top = 0
    __offset_y = 0
    __offset_x = 0
    __chr_list = 'abcdefghimorstuvwxyzABCDEFGHIMORSTUVWXYZ0123456789'
    __hot_key = False

    def __init__(self, name, items, parent, **args):
        self.__index = 0
        self.__name = name
        self.items = items
        self.parent = parent
        self.__start_idx = 0
        self.__stop_idx = 0
        s = list(self.__chr_list)
        shuffle(s)
        self.__chr_list = ''.join(s)
        self.__offset_y = args.get('offset_y', self.__offset_y)
        self.__offset_x = args.get('offset_x', self.__offset_x)
        self.__hot_key = args.get('hot_key', self.__hot_key)

        maxy, maxx = parent.getmaxyx()
        width = max(len(name) + 6, max([len(it) + 5 for it in items]))
        height = min(len(items) + 2, maxy - self.__offset_y)
        Widget.__init__(self, height, width, parent)
        self.resize(height, width)
        y = (maxy - height - self.__offset_y) / 2
        x = (maxx - width - self.__offset_x) / 2
        self.mvwin(self.__offset_y + y, self.__offset_x + x)
        self.move(self.__offset_y + y, self.__offset_x + x)
        self.win().nodelay(1)
        self.win().timeout(0)
        self.win().keypad(1)
        self.__stop_idx = self.__start_idx + self.height() - 2

    def __do_key(self):
        ch = self.win().getch()
        stop = False
        if self.__fm is not None and ch != -1:
            self.__fm()
            self.__fm = None
        if ch in [ ord('j'), ord('\t'), ord('n'), ord('J'), ord('N'), curses.KEY_DOWN ]:
            if self.__index == len(self.items) - 1:
                self.__start_idx = 0
                self.__stop_idx = self.__start_idx + self.height() - 2
                self.__index = 0
            elif self.__index == self.__stop_idx - 1:
                self.__start_idx = self.__start_idx + 1
                self.__stop_idx =  self.__stop_idx + 1
                self.__index = self.__index + 1
            else:
                self.__index = self.__index + 1
        elif ch in [ ord('k'), ord('K'), ord('p'), ord('P'), curses.KEY_UP ]:
            if self.__index == 0:
                self.__start_idx = max(0, len(self.items) - self.height() + 2)
                self.__stop_idx = self.__start_idx + self.height() - 2
                self.__index = len(self.items) - 1
            elif self.__index == self.__start_idx:
                self.__start_idx = self.__start_idx - 1
                self.__stop_idx =  self.__stop_idx - 1
                self.__index = self.__index - 1
            else:
                self.__index = self.__index - 1
        elif ch in [ ord('\n'), ord(' ') ]:
            stop = True
        elif ch in [ ord('q'), ord('Q') ]:
            self.__index = -1
            stop = True
        elif not self.__hot_key:
            pass
        elif ch in [ ord(c) for c in self.__chr_list ]:
            self.__index = [ ord(c) for c in self.__chr_list ].index(ch) + self.__start_idx
            stop = True
        return stop

    def __redraw(self):
        self.win().erase()
        self.win().attron(curses.color_pair(1))
        self.win().box()
        self.win().attroff(curses.color_pair(1))
        self.win().addstr(0, 2, ' %s ' % self.__name, curses.color_pair(3))
        for idx, item in enumerate(self.items):
            if idx < self.__start_idx or idx >= self.__stop_idx:
                continue

            if not self.__hot_key:
                line = item.center(self.width() - 2)
            elif idx - self.__start_idx < len(self.__chr_list):
                pref = ' %s)' % self.__chr_list[idx - self.__start_idx]
                line = pref + item.center(self.width() - 5)
            else:
                pref = '   '
                line = pref + item.center(self.width() - 5)
            flag = 0
            if idx == self.__index:
                flag = curses.A_BOLD | curses.color_pair(5)
            self.win().addstr(idx - self.__start_idx + self.__padding_top + 1, 1, item, flag)

    def run(self, first_movement=None):
        self.__fm = first_movement
        res = True
        while (1):
            if self.__do_key():
                self.win().erase()
                self.win().refresh()
                return self.__index
            self.__redraw()
            self.win().refresh()


class ColumnCheckBox(Widget):
    __padding_left = 5
    __padding_top = 0
    __margin_top = 4
    __margin_bottom = 4
    __offset_y = 0
    __offset_x = 0

    def __init__(self, name, items, parent, **args):
        self.__index = 0
        self.__name = name
        self.items = items
        self.parent = parent
        self.__start_idx = 0
        self.__stop_idx = 0

        self.__margin_top = args.get('marging_top', self.__margin_top)
        self.__margin_bottom = args.get('marging_bottom', self.__margin_bottom)
        self.__offset_y = args.get('offset_y', self.__offset_y)
        self.__offset_x = args.get('offset_x', self.__offset_x)
        maxy, maxx = parent.getmaxyx()
        width = max(len(name) + 6, max([len(str(it)) + 9 for it in items]))
        height = min(len(items) + 2, maxy - self.__offset_y - self.__margin_top - self.__margin_bottom)
        Widget.__init__(self, height, width, parent)
        self.resize(height, width)
        y = (maxy - height - self.__offset_y) / 2
        x = (maxx - width - self.__offset_x) / 2
        self.mvwin(self.__offset_y + y, self.__offset_x + x)
        self.move(self.__offset_y + y, self.__offset_x + x)
        self.win().nodelay(1)
        self.win().timeout(0)
        self.win().keypad(1)
        self.__stop_idx = self.__start_idx + self.height() - 2

    def __do_key(self):
        ch = self.win().getch()
        stop = False
        if self.__fm is not None and ch != -1:
            self.__fm()
            self.__fm = None
        if ch in [ ord('j'), ord('\t'), ord('n'), ord('J'), ord('N'), curses.KEY_DOWN ]:
            if self.__index == len(self.items) - 1:
                self.__start_idx = 0
                self.__stop_idx = self.__start_idx + self.height() - 2
                self.__index = 0
            elif self.__index == self.__stop_idx - 1:
                self.__start_idx = self.__start_idx + 1
                self.__stop_idx =  self.__stop_idx + 1
                self.__index = self.__index + 1
            else:
                self.__index = self.__index + 1
        elif ch in [ ord('k'), ord('K'), ord('p'), ord('P'), curses.KEY_UP ]:
            if self.__index == 0:
                self.__start_idx = max(0, len(self.items) - self.height() + 2)
                self.__stop_idx = self.__start_idx + self.height() - 2
                self.__index = len(self.items) - 1
            elif self.__index == self.__start_idx:
                self.__start_idx = self.__start_idx - 1
                self.__stop_idx =  self.__stop_idx - 1
                self.__index = self.__index - 1
            else:
                self.__index = self.__index - 1
        elif ch in [ ord(' ') ]:
            self.items[self.__index].enable(not self.items[self.__index].enable())
        elif ch in [ ord('\n') ]:
            stop = True
        elif ch in [ ord('q'), ord('Q') ]:
            self.__index = -1
            stop = True
        return stop

    def __redraw(self):
        self.win().erase()
        self.win().attron(curses.color_pair(1))
        self.win().box()
        self.win().attroff(curses.color_pair(1))
        self.win().addstr(0, 2, ' %s ' % self.__name, curses.color_pair(3))
        for idx, item in enumerate(self.items):
            if idx < self.__start_idx or idx >= self.__stop_idx:
                continue
            pref = ' [%s]' % (item.enable() and 'X' or ' ')
            line = pref + str(item).center(self.width() - 6)
            flag = 0
            if idx == self.__index:
                flag = curses.A_BOLD | curses.color_pair(5)
            self.win().addstr(idx - self.__start_idx + self.__padding_top + 1, 1, line, flag)

    def run(self, first_movement=None):
        self.__fm = first_movement
        res = True
        while (1):
            if self.__do_key():
                self.win().erase()
                self.win().refresh()
                return self.__index
            self.__redraw()
            self.win().refresh()
        if self.__index == -1:
            return []
        else:
            return self.items


class InputBox(Widget):
    def __init__(self, parent, prompt="Password", password=False, width=30):
        self.__index = 0
        self.__name = "name"
        self.__offset_x = 0
        self.__offset_y = 0
        self.__password = password
        height = 3
        win = curses.newwin(0, 0, 0, 0)
        Widget.__init__(self, height, width, win, True)
        self.resize(height, width)
        maxy, maxx = parent.getmaxyx()
        y = (maxy - height - self.__offset_y) / 2
        x = (maxx - width - self.__offset_x) / 2
        self.mvwin(y, x)
        win.box()
        win.refresh()
        self.__prompt = prompt + ": "
        win.addstr(1, 1, self.__prompt)
        win.move(1, 1 + len(self.__prompt))
        self.__textbox = curses.textpad.Textbox(win)
        self.__result = ""

    def validator(self, ch):
        y, x = self.win().getyx()
        if ch == 127:
            new_x = max(x - 1, len(self.__prompt) + 1)
            self.win().delch(y, new_x)
            self.win().insch(' ')
            self.win().move(y, new_x)
            self.__result = self.__result[:-1]
            return 0
        elif ch == 7 or ch == 10:                             # submit
            return 7
        elif x == self.width() - 2:
            return 0
        elif ch < 256:
            self.__result += chr(ch)
            if self.__password:
                return ord('*')
            else:
                return ch

    def run(self):
        curses.noecho()
        curses.curs_set(1)
        self.__textbox.edit(self.validator)
        curses.curs_set(0)
        return self.__result


class GalleryPage(Page):
    def __add_widgets(self):
        tidf = lambda: str(oceanbase.get_current_tenant())
        DEBUG(self.__add_widgets, "cur_tenant_id", tidf())
        time_widget = ColumnWidget("TIME-TENANT", [
            Column("timestamp", lambda stat:
                stat["timestamp"].strftime("%H:%m:%S"),
                    10, True),
            Column("tenant", lambda stat:
                tidf(),
                10, True
                ),
            ], self.win())
        sql_count_widget = ColumnWidget("SQL COUNT", [
            Column("sel", lambda stat:
                    sum([item["sql select count"]
                             for item in stat[tidf()].values()]),
                    6, True),
            Column("ins", lambda stat:
                    sum([ item["sql insert count"]
                             for item in stat[tidf()].values() ]),
                    6, True),
            Column("rep", lambda stat:
                    sum([ item["sql replace count"]
                             for item in stat[tidf()].values() ]),
                    6, True),
            Column("del", lambda stat:
                    sum([ item["sql delete count"]
                             for item in stat[tidf()].values() ]),
                    6, True),
            Column("upd", lambda stat:
                    sum([ item["sql update count"]
                             for item in stat[tidf()].values() ]),
                    6, True),
            Column("cmt", lambda stat:
                    sum([ item["trans commit count"]
                             for item in stat[tidf()].values() ]),
                    6, True),
            Column("fail", lambda stat:
                    sum([ item["sql fail exec count"] + item["sql fail query count"]
                             for item in stat[tidf()].values() ]),
                    6, True)
            ], self.win())
        sql_rt_widget = ColumnWidget("SQL RT", [
            Column("select", lambda stat:
                (sum([ item["sql select time"] for item in stat[tidf()].values() ])
                 / float(sum([ item["sql select count"] for item in stat[tidf()].values() ]) or 1) / 1000),
                8),
            Column("insert", lambda stat:
                (sum([ item["sql insert time"] for item in stat[tidf()].values() ])
                 / float(sum([ item["sql insert count"] for item in stat[tidf()].values() ]) or 1) / 1000),
                8),
            Column("update", lambda stat:
                (sum([ item["sql update time"] for item in stat[tidf()].values() ])
                 / float(sum([ item["sql update count"] for item in stat[tidf()].values() ]) or 1) / 1000),
                8),
            Column("replace", lambda stat:
                (sum([ item["sql replace time"] for item in stat[tidf()].values() ])
                 / float(sum([ item["sql replace count"] for item in stat[tidf()].values() ]) or 1) / 1000),
                8),
            Column("delete", lambda stat:
                (sum([ item["sql delete time"] for item in stat[tidf()].values() ])
                 / float(sum([ item["sql delete count"] for item in stat[tidf()].values() ]) or 1) / 1000),
                8),
            ], self.win())
        net_rt_widget = ColumnWidget("NET RT", [
            Column("net", lambda stat:
                    (sum([ item["rpc net delay"] for item in stat[tidf()].values() ])
                     / float(sum([ item["rpc packet in"] for item in stat[tidf()].values() ]) or 1) / 1000),
                    8),
            Column("frame", lambda stat:
                    (sum([ item["rpc net frame delay"] for item in stat[tidf()].values() ])
                     / float(sum([ item["rpc packet in"] for item in stat[tidf()].values() ]) or 1) / 1000),
                    8),
            ], self.win())
        memory_widget = ColumnWidget("MEMORY", [
            Column("active", lambda stat:
                    mem_str(sum([item["active memstore used"] for item in stat[tidf()].values()])),
                8),
            Column("total", lambda stat:
                    mem_str(sum([item["total memstore used"] for item in stat[tidf()].values()])),
                8),
            Column("freeze", lambda stat:
                    mem_str(sum([item["major freeze limit"] for item in stat[tidf()].values()])),
                8),
            Column("limit", lambda stat:
                    mem_str(sum([item["memstore limit"] for item in stat[tidf()].values()])),
                8)
            ], self.win())
        tableapi_widget = ColumnWidget("TABLE API ROWS/SEC", [
            Column("get", lambda stat:
                   (sum([ item["multi retrieve rows"]
                          for item in stat[tidf()].values() ])
                    + sum([ item["single retrieve execute count"]
                            for item in stat[tidf()].values() ])),
                   6, True),
            Column("put", lambda stat:
                   (sum([ item["multi insert_or_update rows"]
                          for item in stat[tidf()].values() ])
                    + sum([ item["single insert_or_update execute count"]
                            for item in stat[tidf()].values() ])),
                   6, True),
            Column("delete", lambda stat:
                   (sum([ item["multi delete rows"]
                          for item in stat[tidf()].values() ])
                    + sum([ item["single delete execute count"]
                            for item in stat[tidf()].values() ])),
                   6, True),
            Column("insert", lambda stat:
                   (sum([ item["multi insert rows"]
                         for item in stat[tidf()].values() ])
                    + sum([ item["single insert execute count"]
                            for item in stat[tidf()].values() ])),
                   6, True),
            Column("update", lambda stat:
                   (sum([ item["multi update rows"]
                          for item in stat[tidf()].values() ])
                    + sum([ item["single update execute count"]
                            for item in stat[tidf()].values() ])),
                   6, True),
            Column("replace", lambda stat:
                   (sum([ item["multi replace rows"]
                          for item in stat[tidf()].values() ])
                    + sum([ item["single replace execute count"]
                            for item in stat[tidf()].values() ])),
                   6, True),
            ], self.win())

        self.add_widget(time_widget)
        self.add_widget(sql_count_widget)
        self.add_widget(sql_rt_widget)
        self.add_widget(net_rt_widget)
        self.add_widget(memory_widget)
        self.add_widget(tableapi_widget)

    def __init__(self, y, x, h, w, parent):
        Page.__init__(self, parent, Layout(), y, x, h, w)
        try:
            self.__add_widgets()
        except curses.error:
            pass

    def title(self):
        return "Gallery"

    def process_key(self, ch):
        if ch == ord('j'):
            pass
        else:
            Page.process_key(self, ch)

DEV = file("debug.log", "w")
def DEBUG(*args):
    global DEV
    if Options.debug:
        print >> DEV, "[%s] %s %s" % ( args[0].func_name, args[1], args[2])

DEVB = open("data.log", "w")
def DEB(*args):
    global DEV
    if Options.debug:
        print >> DEVB, " [%s] %s" % ( args[0].func_name, args[1])

class SQLPage(Page):
    def update_widgets(self):
        def observer_info(stat, ip):
            def try_add(name, key, wrapper):
                try:
                    result[name] = wrapper(stat[oceanbase.get_current_tenant()][ip][key])
                except Exception as e:
#                   DEBUG(try_add, "exception", e)
                    pass
            def try_add2(name, key1, key2, wrapper):
                try:
                    result[name] = wrapper(stat[oceanbase.get_current_tenant()][ip][key1] / (stat[oceanbase.get_current_tenant()][ip][key1] + stat[oceanbase.get_current_tenant()][ip][key2]) or 1)
                    DEBUG(try_add2, "wrapper", (stat[oceanbase.get_current_tenant()][ip][key1] + stat[oceanbase.get_current_tenant()][ip][key2]))
                    DEBUG(try_add2, "wrapper", stat[oceanbase.get_current_tenant()][ip][key1])
                except Exception as e:
                    DEBUG(try_add2, "exception", e)
                    pass
            result = dict()
            try_add("active sessions", "active sessions", count_str)
            try_add2("R-cache hit", "row cache hit", "row cache miss", percent_str)
            try_add2("L-cache hit", "location cache hit", "location cache miss", percent_str)
            try_add2("B-cache hit", "block cache hit", "block cache miss", percent_str)
            try_add2("BI-cache hit", "block index cache hit", "block index cache miss", percent_str)
            try_add2("BloomFilter cache hit", "bloom filter cache hit", "bloom filter cache miss", percent_str)
            return result

        svr = "observer"
        self.clear_widgets()
        DEBUG(self.update_widgets, "tenant.svr_list", oceanbase.get_tenant_svr())
        for ms in oceanbase.get_tenant_svr()[svr]:
            ip = ms['ip']
            port = ms['port']
            DEBUG(observer_info, "oceanbase.get_current_tenant()", oceanbase.get_current_tenant())
            ipport = ip + ":" + port
            f = ColumnFactory(svr, ipport)
            DEBUG(self.update_widgets, "f", f)
            widget = HeaderColumnWidget(
                '%s:%d' % (ip,int(port)),
                SQLPage.generate_columns(f),
                self.win(),
                get_header=lambda stat,ip=ip: observer_info(stat,ipport))
            self.add_widget(widget)

    def __init__(self, y, x, h, w, parent):
        Page.__init__(self, parent, Layout(), y, x, h, w)
        try:
            self.update_widgets()
        except curses.error:
            pass

    def select_columns(self):
        if len(self.all_widgets()) > 0:
            all_widgets = [hc_widget.column_widget() for hc_widget in self.all_widgets()]
            columns = all_widgets[0].valid_columns()
            i = ColumnCheckBox("Select Columns", columns, self.parent()).run()
            for w in all_widgets:
                for idx in range(0, len(columns)):
                    w.valid_columns()[idx].enable(columns[idx].enable())
                w.update()
            [hc_widget.resize() for hc_widget in self.all_widgets()]
            self.rearrange()

    def enable_column_group(self, items):
        # enable table api
        if len(self.all_widgets()) > 0:
            all_widgets = [hc_widget.column_widget() for hc_widget in self.all_widgets()]
            columns = all_widgets[0].valid_columns()
            DEBUG(self.enable_column_group, "column count", len(columns))
            enable_columns = []
            for idx in range(0, len(columns)):
                enable_columns.append(0)
            for idx in items:
                enable_columns[idx] = 1
                DEBUG(self.enable_column_group, "enable idx", idx)
            for w in all_widgets:
                for idx in range(0, len(columns)):
                    w.valid_columns()[idx].enable(enable_columns[idx]==1)
                w.update()
            [hc_widget.resize() for hc_widget in self.all_widgets()]
            self.rearrange()

    def process_key(self, ch):
        if (len(self.shown_widgets())) > 0:
            w = self.selected_widget()
        else:
            return
        if ch == ord('m'):
            curses.endwin()
            oceanbase.mysql()
            curses.doupdate()
        elif ch == ord('O') or ch == ord('o'):
            like_str = InputBox(self.win(), prompt="LIKE STR").run()
            oceanbase.show_sql_result("select name,value from __all_virtual_sys_parameter_stat where svr_ip='%s' and svr_type='mergeserver' and name like '%s' order by name" % (w.host(), like_str))
        elif Options.env == 'online' and ch == ord('l'):
            cmd = "ssh -t %s 'less oceanbase/log/mergeserver.log'" % w.host()
            curses.endwin()
            os.system(cmd)
            curses.doupdate()
        elif ch == ord("="):
            self.select_columns()
        elif ch == ord('A') or ch == ord('a'):
            # table api QPS
            self.enable_column_group([16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38])
        elif ch == ord('B') or ch == ord('b'):
            # table api RT
            self.enable_column_group([17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39])
        elif ch == ord('S') or ch == ord('s'):
            # SQL
            self.enable_column_group(range(13))
        else:
            Page.process_key(self, ch)

    def title(self):
        return "ServerPage"

    @staticmethod
    def generate_columns(f):
        enable_sql_columns = True
        enable_table_api_columns = False
        enable_table_api_rt_columns = False
        return [
            #default(SQL)
            f.count("Sql Select Count", "ssc", "sql select count", enable=enable_sql_columns),
            f.time("Sql Select Time", "ssrt", "sql select time", "sql select count", enable=enable_sql_columns),
            f.count("Sql Update Count", "suc", "sql update count", enable=enable_sql_columns),
            f.time("Sql Update Time", "surt", "sql update time", "sql update count", enable=enable_sql_columns),
            f.count("Sql Insert Count", "sic", "sql insert count", enable=enable_sql_columns),
            f.time("Sql Insert Time", "sirt", "sql insert time", "sql insert count", enable=enable_sql_columns),
            f.count("Sql Delete Count", "sdc", "sql delete count", enable=enable_sql_columns),
            f.time("Sql Delete Time", "sdrt", "sql delete time", "sql delete count", enable=enable_sql_columns),
            f.count("Trans Commit Count", "tcc", "trans commit count", enable=enable_sql_columns),
            f.time("Trans Commit Time", "tcrt", "trans commit time", "trans commit count", enable=enable_sql_columns),
            f.count("Sql Execute Local Count", "selc", "sql local count", enable=enable_sql_columns),
            f.count("Sql Execute Remote Count", "serc", "sql remote count", enable=enable_sql_columns),
            f.count("Sql Execute Distributed Count", "sedc", "sql distributed count", enable=enable_sql_columns),
            f.count("Inner SQL Connection Execute Count", "iscec", "inner sql connection execute count"),
            f.time("Inner SQL Connection Execute Time", "iscet", "inner sql connection execute time", "inner sql connection execute count"),
            # Table API
            f.count("Table API table api login count", "login", "table api login count"),

            f.count("Table API single retrieve execute count", "get", "single retrieve execute count", enable=enable_table_api_columns),
            f.time("Table API single retrieve execute time", "g-rt", "single retrieve execute time", "single retrieve execute count", enable=enable_table_api_rt_columns),
            f.count("Table API single insert execute count", "insert", "single insert execute count", enable=enable_table_api_columns),
            f.time("Table API single insert execute time", "i-rt", "single insert execute time", "single insert execute count", enable=enable_table_api_rt_columns),
            f.count("Table API single delete execute count", "delete", "single delete execute count", enable=enable_table_api_columns),
            f.time("Table API single delete execute time", "d-rt", "single delete execute time", "single delete execute count", enable=enable_table_api_rt_columns),
            f.count("Table API single update execute count", "update", "single update execute count", enable=enable_table_api_columns),
            f.time("Table API single update execute time", "u-rt", "single update execute time", "single update execute count", enable=enable_table_api_rt_columns),
            f.count("Table API single insert_or_update execute count", "put", "single insert_or_update execute count", enable=enable_table_api_columns),
            f.time("Table API single insert_or_update execute time", "p-rt", "single insert_or_update execute time", "single insert_or_update execute count", enable=enable_table_api_rt_columns),
            f.count("Table API single replace execute count", "replace", "single replace execute count", enable=enable_table_api_columns),
            f.time("Table API single replace execute time", "r-rt", "single replace execute time", "single replace execute count", enable=enable_table_api_rt_columns),

            f.count("Table API multi retrieve execute count", "mget", "multi retrieve execute count", enable=enable_table_api_columns),
            f.time("Table API multi retrieve execute time", "mg-rt", "multi retrieve execute time", "multi retrieve execute count", enable=enable_table_api_rt_columns),
            f.count("Table API multi insert execute count", "minsert", "multi insert execute count", enable=enable_table_api_columns),
            f.time("Table API multi insert execute time", "mi-rt", "multi insert execute time", "multi insert execute count", enable=enable_table_api_rt_columns),
            f.count("Table API multi delete execute count", "mdelete", "multi delete execute count", enable=enable_table_api_columns),
            f.time("Table API multi delete execute time", "md-rt", "multi delete execute time", "multi delete execute count", enable=enable_table_api_rt_columns),
            f.count("Table API multi update execute count", "mupdate", "multi update execute count", enable=enable_table_api_columns),
            f.time("Table API multi update execute time", "mu-rt", "multi update execute time", "multi update execute count", enable=enable_table_api_rt_columns),
            f.count("Table API multi insert_or_update execute count", "mput", "multi insert_or_update execute count", enable=enable_table_api_columns),
            f.time("Table API multi insert_or_update execute time", "mp-rt", "multi insert_or_update execute time", "multi insert_or_update execute count", enable=enable_table_api_rt_columns),
            f.count("Table API multi replace execute count", "mreplace", "multi replace execute count", enable=enable_table_api_columns),
            f.time("Table API multi replace execute time", "mr-rt", "multi replace execute time", "multi replace execute count", enable=enable_table_api_rt_columns),
            f.count("Table API batch retrieve execute count", "bget", "batch retrieve execute count"),
            f.time("Table API batch retrieve execute time", "bg-rt", "batch retrieve execute time", "batch retrieve execute count"),
            f.count("Table API batch hybrid execute count", "bhybrid", "batch hybrid execute count", enable=enable_table_api_columns),
            f.time("Table API batch hybrid execute time", "bh-rt", "batch hybrid execute time", "batch hybrid execute count", enable=enable_table_api_rt_columns),

            f.count("Table API multi retrieve rows", "rget", "multi retrieve rows"),
            f.time("Table API multi retrieve execute time/row", "rg-rt", "multi retrieve execute time", "multi retrieve rows"),
            f.count("Table API multi insert rows", "rinsert", "multi insert rows"),
            f.time("Table API multi insert execute time/row", "ri-rt", "multi insert execute time", "multi insert rows"),
            f.count("Table API multi delete rows", "rdelete", "multi delete rows"),
            f.time("Table API multi delete execute time/row", "rd-rt", "multi delete execute time", "multi delete rows"),
            f.count("Table API multi update rows", "rupdate", "multi update rows"),
            f.time("Table API multi update execute time/row", "ru-rt", "multi update execute time", "multi update rows"),
            f.count("Table API multi insert_or_update rows", "rput", "multi insert_or_update rows"),
            f.time("Table API multi insert_or_update execute time", "rp-rt", "multi insert_or_update execute time", "multi insert_or_update rows"),
            f.count("Table API multi replace rows", "rreplace", "multi replace rows"),
            f.time("Table API multi replace execute time/row", "rr-rt", "multi replace execute time", "multi replace rows"),
            f.count("Table API batch retrieve rows", "rbget", "batch retrieve rows"),
            f.time("Table API batch retrieve execute time/row", "rbg-rt", "batch retrieve execute time", "batch retrieve rows"),
            f.count("Table API batch hybrid rows", "rbhybrid", "batch hybrid insert_or_update rows"),
            f.time("Table API batch hybrid execute time/row", "rbh-rt", "batch hybrid execute time", "batch hybrid insert_or_update rows"),

            #inner packet
            f.count("RPC Packet In", "rpci", "rpc packet in"),
            f.count("RPC Packet In Bytes", "rpci(B)", "rpc packet in bytes"),
            f.count("RPC Packet Out", "rpco", "rpc packet out"),
            f.count("RPC Packet Out Bytes", "rpco(B)", "rpc packet out bytes"),
            f.count("RPC Deliver Fail", "rpc fail", "rpc deliver fail"),
            f.time("RPC Net Delay", "rpc delay", "rpc net delay"),
            f.count("RPC Net Frame Delay", "rpc frame delay", "rpc net frame delay"),
            #outer packet
            f.count("MySQL Packet In", "mpci", "mysql packet in"),
            f.count("MySQL Packet In Bytes", "mpci(B)", "mysql packet in bytes"),
            f.count("MySQL Packet Out", "mpco", "mysql packet out"),
            f.count("MySQL Packet Out Bytes", "mpco(B)", "mysql packet out bytes"),
            f.count("MySQL Deliver Fail", "mdf", "mysql deliver fail"),
            #request queue
            f.count("Request Enqueue Count", "enqueue", "request enqueue count"),
            f.count("Request Dequeue Count", "dequeue", "request dequeue count"),
            f.time("Request Queue Time", "QT", "request queue time"),
            #transmition
            f.time("Trans Commit Log Time", "tclt", "trans commit log time"),
            f.count("Trans Commit Log Sync Count", "tclsc(sync)", "trans commit log sync count"),
            f.count("Trans Commit LOg Submit Count", "tclsc(submit)", "trans commit log submit count"),
            f.count("Trans System Trans Count", "tstc", "trans system trans count"),
            f.count("Trans User Trans Count", "tutc", "trans user trans count"),
            f.count("Trans Start Count", "tsc", "trans start count"),
            f.count("Trans Total Used Time", "ttut", "trans total used time"),
            f.count("Trans Commit Count", "tcc", "trans commit count"),
            f.time("Trans Commit Time", "tct", "trans commit time"),
            f.count("Trans Rollback Count", "trc", "trans rollback count"),
            f.time("Trans Rollback Time", "trt", "trans rollback time"),
            f.count("Trans Timeout Count", "ttc", "trans timeout count"),
            f.count("Trans Single Partition Count", "tspc", "trans single partition count"),
            #cache
            #FIXME
            f.count("Row Cache Hit", "rch", "row cache hit"),
            f.count("Row Cache Miss", "rcm", "row cache miss"),
            f.count("Block Index Cache Hit", "bich", "block index cache hit"),
            f.count("Block Index Cache Miss", "bicm", "block index cache miss"),
            f.count("Bloom Filter Cache Hit", "bfch", "bloom filter cache hit"),
            f.count("Bloom Filter Cache Miss", "bfcm", "bloom fileter cache miss"),
            f.count("Bloom Filter Filts", "bff", "bloom filter filts"),
            f.count("Bloom Filter Passes", "bfp", "bloom filter passes"),
            f.count("Block Cache Hit", "bch", "block cache hit"),
            f.count("Block Cache Miss", "bcm", "block cache miss"),
            f.count("Location Cache Hit", "lch", "location cache hit"),
            f.count("Location Cache Miss", "lcm", "location cache miss"),
            #resources
            f.count("Active Sessions", "as", "active sessions"),
            f.count("IO Read Count", "iorc", "io read count"),
            f.count("IO Read Delay", "iord", "io read delay"),
            f.count("IO Read Bytes", "ior(B)", "io read bytes"),
            f.count("IO Write Count", "iowc", "io write count"),
            f.count("IO Write Delay", "iowd", "io write delay"),
            f.count("IO Write Bytes", "iow(B)", "io write delay"),
            f.count("Memstore Scan Count", "msc", "memstore scan count"),
            f.count("Memstore Scan Succ Count", "mssc", "memstore scan succ count"),
            f.count("Memstore Scan Fail Count", "msfc", "memstore scan fail count"),
            f.count("Memstore Get Count", "mgc", "memstore get count"),
            f.count("Memstore Get Succ Count", "mgsc", "memstore get succ count"),
            f.count("Memstore Get Fail Count", "mgfc", "memstore get fail count"),
            f.count("Memstore Apply Count", "mac", "memstore apply count"),
            f.count("Memstore Apply Succ Count", "masc", "memstore apply succ count"),
            f.count("Memstore Apply Fail Count", "mafc", "memstore apply fail count"),
            f.count("Memstore Row Count", "mrc", "memstore row count"),
            f.time("Memstore Get Time", "mgt", "memstore get time"),
            f.time("Memstore Scan Time", "mst", "memstore scan time"),
            f.time("Memstore Apply Time", "mat", "memstore apply time"),
            f.count("Memstore Read Lock Succ Count", "mrlsc", "memstore read lock succ count"),
            f.count("Memstore Read Lock Fail Count", "mrlfc", "memstore read lock fail count"),
            f.count("Memstore Write Lock Succ Count", "mwlsc", "memstore write lock succ count"),
            f.count("Memstore Write Lock Fail Count", "mwlfc", "memstore write lock fail count"),
            f.time("Memstore Wait Write Lock Time", "mwwlt", "memstore wait write lock time"),
            f.time("Memstore Wait Read Lock Time", "mwrlt", "memstore wait read lock time"),
            f.count("IO Read Micro Index Count", "iormic", "io read micro index count"),
            f.count("IO Read Micro Index Bytes", "iormib", "io read micro index bytes"),
            f.count("IO Prefetch Micro Block Count", "iopmbc", "io prefetch micro block count"),
            f.count("IO Prefetch Micro Block Bytes", "iopmbb", "io prefetch micro block bytes"),
            f.count("IO Read Uncompress Micro Block Count", "iorumbc", "io read uncompress micro block count"),
            f.count("IO Read Uncompress Micro Block Bytes", "iorumbb", "io read uncompress micro block bytes"),
            f.count("Active Memstore Used", "amu", "active memstore used"),
            f.count("Total Memstore Used", "tmu", "total memstore used"),
            f.count("Major Freeze Trigger", "mft", "major freeze trigger"),
            f.count("Memstore Limit", "ml", "memstore limit"),
            f.count("Min Memory Size", "mms(min)", "min memory size"),
            f.count("Max Memory Size", "mms(max)", "max memory size"),
            f.count("Memory Usage", "mu", "memory usage"),
            f.count("Min CPUS", "mc(min)", "min cpus"),
            f.count("Max CPUS", "mc(max)", "max cpus"),
            #meta
            f.count("Refresh Schema Count", "rsc", "refresh schema count"),
            f.time("Refresh Schema Time", "rst", "refresh schema time"),
            f.count("Partition Table Operator Get Count", "ptogc", "partition table operator get count"),
            f.time("Partition Table Operator Get Time", "ptogt", "partition table operator get time"),
            #clog
            f.count("Submitted To Sliding Window Log Count", "sswlc", "submitted to sliding window log count"),
            f.count("Submitted To Sliding Window Log Size", "sswls", "submitted to sliding window log size"),
            f.count("Index Log Flushed Count", "ilfc", "index log flushed count"),
            f.count("Index Log Flushed Clog Size", "ilfcs", "index log flushed clog size"),
            f.count("Clog Flushed Count", "cfc", "clog flushed count"),
            f.count("Clog Flushed Size", "cfs", "clog flushed size"),
            f.count("Clog Read Count", "crc(read)", "clog read count"),
            f.count("Clog Read Size", "crs", "clog read size"),
            f.count("Clog Disk Read Size", "cdrs", "clog disk read size"),
            f.count("Clog Disk Read Count", "cdrc", "clog disk read count"),
            f.time("Clog Disk Read Time", "cdrt", "clog disk read time"),
            f.count("Clog Fetch Log Size", "cfls", "clog fetch log size"),
            f.count("Clog Fetch Log Count", "cflc", "clog fetch log size"),
            f.count("Clog Fetch Log By Location Size", "cflbls", "clog fetch log by location size"),
            f.count("Clog Fetch Log By Location Count", "cflblc", "clog fetch log by location count"),
            f.count("Clog Read Request Succ Size", "crrss", "clog read request succ size"),
            f.count("Clog Read Request Succ Count", "crrsc", "clog read request succ count"),
            f.count("Clog Read Request Fail Count", "crrfc", "clog read request fail count"),
            f.time("Clog Confirm Delay Time", "ccdt", "clog confirm delay time"),
            f.count("Clog Flush Task Generate Count", "cftgc", "clog flush task generate count"),
            f.count("Clog Flush Task Release Count", "cftrc", "clog flush task release count"),
            f.count("Clog RPC Delay Time", "crdt", "clog rpc delay time"),
            f.count("Clog RPC Count", "crc(rpc)", "clog rpc count"),
            f.count("Clog Non KV Cache Hit Count", "cnkchc", "clog non kv cache hit count"),
            f.time("Clog RPC Request Handle Time", "crrht", "clog rpc request handle time"),
            f.count("Clog RPC Request Count", "crrc", "clog rpc request count"),
            f.count("Clog Cache Hit Count", "cchc", "clog cache hit count")]

class MachineStatPage(Page):
    def update_widgets(self):
        pass

    def title(self):
        return "Machine Stat"

    def __init__(self, y, x, h, w, parent):
        Page.__init__(self, parent, Layout(), y, x, h, w)

class HelpPage(Page):
    def __init__(self, y, x, h, w, parent):
        Page.__init__(self, parent, Layout(), y, x, h, w)
        self.win().bkgd(curses.color_pair(4))

    def redraw(self):
        nline = [0]                       # hack mutating outer variables for python2
        def addline(x, line, attr=0):
            self.win().addstr(nline[0], x, line, attr)
            nline[0] += 1
        def addkeys(keys):
            for key_item in keys:
                if 0 == len(key_item):
                    string = ""
                else:
                    string = "    %-14s: %s" % key_item
                addline(4, string)
            nline[0] += 1
        def addgroup(group_name, keys):
            addline(4, group_name, curses.color_pair(4) | curses.A_BOLD)
            addline(2, "----------------------------------------------", curses.color_pair(4) | curses.A_BOLD)
            addkeys(keys)
        try:
            Page.redraw(self)
            ob_keys = [('c', 'Switch between tenants')]
            addgroup("Global Keys  -  OceanBase", ob_keys)
            widget_keys = [('Tab','Select next widget'),
                           ('m', 'Connect to oceanbase using mysql client'),
                           ('j', 'ssh to selected host')]
            addgroup("Global Keys  -  Widget", widget_keys)
            pages_keys = [('1 F1', 'Help page'), ('2 F2', 'Gallery page'), ('3 F3', 'Observer page'),
                          ('d', 'Delete selected widget'), ('R', 'Restore deleted widgets'),
                          ('=', 'Filter Columns for ObServer Page'),
                          ('a A', 'Show columns for Table API QPS'),
                          ('b B', 'Show columns for Table API RT'),
                          ('s S', 'Show columns for SQL QPS and RT'),
            ]
            addgroup("Global Keys  -  Page", pages_keys)
            test_keys = [('p', 'Messsage box tooltips')]
            addgroup("Global Keys  -  Test", test_keys)
            select_keys = [('DOWN TAB J N', 'Next item'), ('UP K P', 'Previous item'),
                           ('SPC ENTER', 'Select current item'),
                           ('Q q', 'Quit selection box')]
            addgroup("Global Keys  -  Selection Box", select_keys)
            system_keys = [('q', 'quit dooba')]
            addgroup("Global Keys  -  System", system_keys)
            support = [
                ('Author', 'oceanbase'),
                ('Mail', ''),
                ()
                ]
            addgroup("Support", support)
        except curses.error:
            pass

    def title(self):
        return 'Help'


class BlankPage(Page):
    def __init__(self, y, x, h, w, parent):
        Page.__init__(self, parent, Layout(), y, x, h, w)

    def title(self):
        return 'Blank Page'

class Dooba(object):
    def build_oceanbase(self):
        return '''
  ___                       ____
 / _ \  ___ ___  __ _ _ __ | __ )  __ _ ___  ___
| | | |/ __/ _ \/ _` | \'_ \|  _ \ / _` / __|/ _ \\
| |_| | (_|  __/ (_| | | | | |_) | (_| \__ \  __/
 \___/ \___\___|\__,_|_| |_|____/ \__,_|___/\___|'''

    def build_dooba(self):
        return '''
     _             _
  __| | ___   ___ | |__   __ _
 / _` |/ _ \ / _ \| \'_ \ / _` |
| (_| | (_) | (_) | |_) | (_| |
 \__,_|\___/ \___/|_.__/ \__,_|'''

    def __init_curses(self, win):
        self.stdscr = win
        self.stdscr.keypad(1)
        self.stdscr.nodelay(1)
        self.stdscr.timeout(0)
        self.maxy, self.maxx = self.stdscr.getmaxyx()
        curses.curs_set(0)
        curses.noecho()
        self.__term = curses.termname()
        self.__init_colors()

    def __init_colors(self):
        if curses.has_colors():
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_RED, -1)   # header widget and status widget
            curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_GREEN) # column header
            curses.init_pair(3, curses.COLOR_MAGENTA, -1) # widget title
            curses.init_pair(4, curses.COLOR_YELLOW, -1) # help page color
            curses.init_pair(5, curses.COLOR_RED, curses.COLOR_CYAN)
            curses.init_pair(6, curses.COLOR_GREEN, -1) # column header
            curses.init_pair(7, curses.COLOR_WHITE, -1) # machine stat header
            # curses.init_pair(8, curses.COLOR_GRAY, -1)  # machine stat text

    def __resize_term(self):
        self.maxy, self.maxx = self.stdscr.getmaxyx()
        try:
            self.help_w.move(0, 0)
            self.help_w.mvwin(0, 0)
            self.help_w.resize(2, self.maxx)
            self.stat_w.move(self.maxy - 2, 0)
            self.stat_w.resize(2, self.maxx)
            for page in self.__all_page:
                page.resize(self.maxy - 4, self.maxx)
        except curses.error:
            pass
        self.stdscr.erase()

    def __do_key(self):
        ch = self.__all_page[self.__page_index].getch()
        if ch == ord('q'):
            curses.endwin()
            return True
        elif ch >= ord('0') and ch <= ord('9'):
            index = ch - ord('0')
            if index > 0 and index < len(self.__all_page):
                self.__page_index = ch - ord('0')
                self.help_w.switch_page(self.__all_page[self.__page_index])
        elif ch >= curses.KEY_F1 and ch <= curses.KEY_F9:
            self.__page_index = ch - curses.KEY_F1
            self.help_w.switch_page(self.__all_page[self.__page_index])
        elif ch == ord('\t'):
            self.__all_page[self.__page_index].select_next()
        elif ch == ord('w'):
            pagename = self.__all_page[self.__page_index].title()
            appname = oceanbase.app
            filename = "%s_%s.dooba.win.bz2" % (appname, pagename)
            tmpf = tempfile.TemporaryFile()
            self.stdscr.putwin(tmpf);
            tmpf.seek(0)
            f = bz2.BZ2File(filename, "w")
            f.write(tmpf.read())
            f.close()
            tmpf.close()
            MessageBox(self.stdscr, "[ INFO ] save screen to %s done!" % filename).run(anykey=True)
        elif ch == ord('p'):
            MessageBox(self.stdscr, "[TEST]What's up? (- -)#").run()
        elif ch == ord('z'):
            items = [('check1  12344', False), ('check2 1', True), ('check3 124328432431', False)]
            CheckBox('Test check box', items, self.stdscr).run()
#       elif ch == ord('i'):
#           result = InputBox(self.stdscr).run()
#           MessageBox(self.stdscr, "[TEST] %s" % result).run(anykey=True)
        elif ch == ord('C'):
            dataid_bk = Options.dataid
            self.stdscr.erase()
            self.__show_logo()
            while True:
                if self.__select_dataid() < 0:
                    Options.dataid = dataid_bk
                    break
                elif self.__update_lms() and self.__select_cluster():
                    break
            if dataid_bk != Options.dataid:
                oceanbase.stop()
                oceanbase.start()
                for page in self.__all_page:
                    page.update()
        elif ch == ord('c'):
            self.__select_cluster()
        elif ch == ord("t"):
            PopPad("Abbreviation", [], self.stdscr).run()
        self.__all_page[self.__page_index].process_key(ch)

    def __run(self):
        while (1):
            self.stdscr.erase()

            self.help_w.redraw()
            self.__all_page[self.__page_index].redraw()
            self.stat_w.redraw()

            if (curses.is_term_resized(self.maxy, self.maxx)):
                self.__resize_term()
            if self.__do_key():
                break

            self.stdscr.refresh()
            sleep(0.05)

    def __select_cluster(self):
        if len(oceanbase.tenant) <= 1:
            idx = 0
        else:
            DEBUG(self.__select_cluster, "tnt.selected", [tnt.selected for tnt in oceanbase.tenant])
            idx = SelectionBox("Select Tenant", [ "[%s] %s " % ("*" if tnt.selected else " ", tnt.tenant_name) for tnt in oceanbase.tenant], self.stdscr).run()
            if True == oceanbase.tenant[idx].selected:
                pass
            else:
                tid = oceanbase.tenant[idx].tenant_id
                tname = oceanbase.tenant[idx].tenant_name
                oceanbase.switch_tenant(tid)
                DEBUG(self.__select_cluster, "global oceanbase", oceanbase)
                DEBUG(self.__select_cluster, "oceanbase.tenant", oceanbase.tenant[idx])
                DEBUG(self.__select_cluster, "oceanbase.tenant[idx]", oceanbase.tenant[idx].selected)
                DEBUG(self.__select_cluster, "oceanbase.get_current_tenant()", oceanbase.get_current_tenant())
                for page in self.__all_page:
                    page.cur_tenant_id = tid
                    page.update()
                DEBUG(self.__select_cluster, "page.cur_tenant_id", [" ".join([str(page.title()), page.cur_tenant_id]) for page in self.__all_page])

    def __show_logo(self):
        self.stdscr.hline(7, 0, curses.ACS_HLINE, 1024, curses.A_BOLD | curses.color_pair(1))
        self.stdscr.refresh()
        oceanbase_width = max([len(line) for line in self.build_oceanbase().split('\n')])
        oceanbase_height = len(self.build_oceanbase().split('\n'))
        ob_win = curses.newwin(oceanbase_height + 1, oceanbase_width + 1, 0, 10)
        ob_win.addstr(self.build_oceanbase(), curses.A_BOLD | curses.color_pair(1))
        ob_win.refresh()

        dooba_width = max([len(line) for line in self.build_dooba().split('\n')])
        dooba_height = len(self.build_dooba().split('\n'))
        dooba_win = curses.newwin(dooba_height + 1, dooba_width + 1, 0, self.maxx - dooba_width - 10)
        dooba_win.addstr(self.build_dooba(), curses.A_BOLD | curses.color_pair(6))
        dooba_win.refresh()

    def __cowsay(self, saying):
        cowsay = str(Cowsay(saying))
        cowsay_width = max([len(line) for line in cowsay.split('\n')])
        cowsay_height = len(cowsay.split('\n'))
        try:
            cowsay_win = curses.newwin(cowsay_height + 1, cowsay_width + 1,
                                   self.maxy - cowsay_height - 2, self.maxx - cowsay_width - 10)
            cowsay_win.addstr(cowsay, curses.color_pair(3))
            cowsay_win.refresh()
            self.__cowsay_win = cowsay_win
        except curses.error:
            pass

    def __search_app(self, search_text, dataid_list):
        ret = []
        for group in dataid_list:
            for dataid in dataid_list[group]:
                if search_text in dataid: ret.append(dataid)
        if not ret:
            ret.append("Could not Find App Name Contains: %s" % search_text)
        return ret

    def __select_dataid(self):
        def clear_cowsay():
            try:
                self.__cowsay_win.erase()
                self.__cowsay_win.refresh()
            except AttributeError:
                pass

        dataid_list = ObConfigHelper().get_dataid_list()
        if len(dataid_list) == 0:
            MessageBox(self.stdcsr, "Can't fetch dataid list, please check your environment").run(anykey=True)
            curses.endwin()
            exit(1)
        idx = SelectionBox("Select OceanBase App Groups", dataid_list, self.stdscr, offset_y=8, hot_key=False).run(first_movement=clear_cowsay)
        tmp_k= ""
        if idx == -2:
            search_text = InputBox(self.stdscr, prompt="App Search").run()
            search_res = self.__search_app(search_text, dataid_list)
            idx = SelectionBox("Searching Results", search_res, self.stdscr, offset_y=8, hot_key=True).run(first_movement=clear_cowsay)
            Options.dataid = search_res[idx]
            oceanbase.update_dataid(Options.dataid)
        else:
            for seq, k in zip(range(len(dataid_list)), dataid_list):
                if seq == idx:
                    tmp_k = k
                    break
            idx = SelectionBox("Select OceanBase Data ID", dataid_list[tmp_k], self.stdscr, offset_y=8, hot_key=True).run(first_movement=clear_cowsay)

            if idx >= 0 and tmp_k in dataid_list.keys():
                Options.dataid = dataid_list[tmp_k][idx]
                oceanbase.update_dataid(Options.dataid)
        return idx

#   def __select_dataid(self):
#       def clear_cowsay():
#           try:
#               self.__cowsay_win.erase()
#               self.__cowsay_win.refresh()
#           except AttributeError:
#               pass

#       dataid_list = ObConfigHelper().get_dataid_list()
#       if len(dataid_list) == 0:
#           MessageBox(self.stdscr, "Can't fetch dataid list, plz check your environment!").run(anykey=True)
#           curses.endwin()
#           exit(1)
#       idx = SelectionBox('Select OceanBase dataID',
#                          dataid_list, self.stdscr, offset_y=8, hot_key=True).run(first_movement=clear_cowsay)
#       if idx >= 0:
#           Options.dataid = dataid_list[idx]
#           oceanbase.update_dataid(Options.dataid)
#       return idx

    def __update_lms(self):
        return oceanbase.check_lms(self.__cowsay)

    def __obssh(self, dataid):
        #oceanbase.update_cluster_info()
        oceanbase.update_tenant_info()
        clusters = oceanbase.cluster
        ipmap = {}
        for clu in clusters:
            for svr in clu.svr_list:
                for s in clu.svr_list[svr]:
                    ip = s['ip']
                    if ip not in ipmap:
                        ipmap[ip] = {}
                    if "server" not in ipmap[ip]:
                        ipmap[ip]["server"] = []
                    ipmap[ip]["id"] = clu.id
                    ipmap[ip]["role"] = clu.role
                    ipmap[ip]["server"].append({'type': svr, 'role': s['role']})
        def get_sign(ipmap):
            name_map = {'rootserver':'R', 'updateserver':'U',
                        'mergeserver':'M', 'chunkserver':'C'}
            sign = ''
            for svr in ipmap["server"]:
                s = name_map[svr['type']]
                if svr['role'] == 2:
                    s = s.lower()
                sign += s
            sign = '<%d:%s>' % (ipmap["id"], ipmap["role"] == 1 and 'MASTR' or 'SLAVE') + ('%4s' % sign).replace(' ', '_')
            return sign
        def sort_cmp(l, r):
            cmp0 = cmp(l[1], r[1])
            cmp1 = cmp(l[0], r[0])
            if cmp0 != 0:
                return cmp0
            else:
                return cmp1

        iplist = sorted([(ip, get_sign(ipmap[ip])) for ip in ipmap.keys()], cmp=sort_cmp)
        idx = SelectionBox("select server (%s)" % dataid,
                           ["   %-16s  |  %14s " % (info[0], info[1]) for info in iplist],
                           self.stdscr, hot_key=True).run()
        if idx >= 0:
            curses.endwin()
            oceanbase.ssh(iplist[idx][0])
            curses.doupdate()
            return True
        return False

    def __run_obssh(self):
        if Options.dataid is None:
            while True:
                if self.__select_dataid() < 0:
                    curses.endwin()
                    exit(0)
                if not self.__update_lms():
                    continue
                while self.__obssh(Options.dataid):
                    pass
        else:
            if not self.__update_lms():
                exit(1)
            while self.__obssh(Options.dataid):
                pass
            else:
                exit(0)

    def __run_show_win_file(self):
        self.stdscr.erase()
        bzfh = bz2.BZ2File(Options.show_win_file)
        fh = tempfile.TemporaryFile()
        fh.write(bzfh.read())
        fh.seek(0)
        bzfh.close()
        self.stdscr = curses.getwin(fh)
        self.stdscr.nodelay(0)
        self.stdscr.notimeout(1)
        fh.close()
        ch = self.stdscr.getch()
        curses.endwin()
        exit(0)

    def __init__(self, win):
        self.__all_page = []
        self.__page_index = 2

        self.__init_curses(win)
        minx = 80
        if self.maxx < minx or self.maxy < 20:
            MessageBox(self.stdscr,
                       "[ ERROR ] %dx%d is tooooo smalllll! %dx20 is at least..." % (self.maxx, self.maxy, minx)).run(anykey=True)
            return
        self.__select_cluster()
        oceanbase.update_tenant_info()
        oceanbase.start()
        self.__show_logo()

       #if Options.show_win_file:
       #    self.__run_show_win_file()
       #elif Options.degradation:
       #    self.__run_obssh()
       #elif Options.using_ip_port:
       #    #oceanbase.update_cluster_info()
       #    oceanbase.update_tenant_info()
       #    self.__select_cluster()
       #    oceanbase.start()
       #else:
       #    if Options.dataid is None:
       #        while True:
       #            if self.__select_dataid() < 0:
       #                curses.endwin()
       #                exit(0)
       #            if self.__update_lms() and self.__select_cluster():
       #                break
       #    elif not self.__update_lms():
       #        curses.endwin()
       #        print ("ERROR: update lms failed for [%s]" % Options.dataid)
       #        exit(1)
       #    elif not self.__select_cluster():
       #        exit(0)
       #    oceanbase.start()

        self.stat_w = StatusWidget(self.stdscr)
        self.help_w = HeaderWidget(self.stdscr)

        self.__all_page.append(BlankPage(2, 0, self.maxy - 4, self.maxx, self.stdscr))        # 0
        self.__all_page.append(HelpPage(2, 0, self.maxy - 4, self.maxx, self.stdscr))         # 1
        self.__all_page.append(GalleryPage(2, 0, self.maxy - 4, self.maxx, self.stdscr))      # 2
        self.__all_page.append(SQLPage(2, 0, self.maxy - 4, self.maxx, self.stdscr))          # 3
        # self.__all_page.append(HistoryPage(2, 0, self.maxy - 4, self.maxx, self.stdscr))      # 4
        # self.__all_page.append(BianquePage(2, 0, self.maxy - 4, self.maxx, self.stdscr))      # 5
        # self.__all_page.append(MachineStatPage(2, 0, self.maxy-4, self.maxx, self.stdscr))    # 6
        # self.__all_page.append(BlankPage(2, 0, self.maxy - 4, self.maxx, self.stdscr))        # 7
        # self.__all_page.append(BlankPage(2, 0, self.maxy - 4, self.maxx, self.stdscr))        # 8
        # self.__all_page.append(BlankPage(2, 0, self.maxy - 4, self.maxx, self.stdscr))        # 9

        self.__run()


class DoobaMain(object):
    '''NAME
        dooba - A curses powerful tool for OceanBase admin, more than a monitor

SYNOPSIS
        dooba [OPTIONS]

OPTIONS

    ·   --host=HOST, -h HOST

        Connect to OceanBase on the given host.

    ·   --port=PORT, -P PORT

        The TCP/IP port to use for connecting to OceanBase server.

    ·   --user=USER, -u USER

        The user to use for connecting to OceanBase server.

    ·   --password=PASSWORD, -p PASSWORD

        The password to use for connecting to OceanBase server.

EXAMPLES

    ./dooba -h127.0.0.1 -P2800 -uroot
    '''

    def __usage(self):
        print('Usage: dooba [-h|--host=HOST] [-P|--port=PORT] [-u|--user=USER] [-p|--password=PASSWORD] [--tenant=TENANT]')
        #print('Usage: dooba --show dooba_win_file')

    def __set_env(self):
        setlocale(LC_ALL, "en_US.UTF-8")
        environ['TERM'] = 'xterm'

    def __parse_options(self):
        try:
            opts, args = getopt(sys.argv[1:], '?dh:i:I:p:P:su:D',
                                ['debug', 'help', 'host=', 'interval=', 'port=',
                                 'password=', 'supermode', 'user=', 'dataid=',
                                 'online', 'offline', 'machine-interval=', 'degradation',
                                 'show=', 'daemon', 'http', 'start', 'stop', 'restart',
                                 'http-port=', 'tenant=', 'tenant_id='])
        except GetoptError as err:
            print str(err) # will print something like "option -a not recognized"
            self.__usage()
            exit(2)
        for o, v in opts:
            if o in ('-?', '--help'):
                print self.__doc__
                exit(1)
            if o in ('-d', '--debug'):
                Options.debug = True
            elif o in ('-h', '--host'):
                Options.host = v
                Options.using_ip_port = True
            elif o in ('-P', '--port'):
                Options.port = int(v)
                Options.using_ip_port = True
            elif o in ('-u', '--user'):
                Options.user = v
            elif o in ('-p', '--password'):
                Options.password = v
            elif o in ('-s', '--supermode'):
                Options.supermode = True
            elif o in ('-i', '--interval'):
                Options.interval = float(v)
            elif o in ('-I', '--machine-interval'):
                Options.machine_interval = float(v)
            elif o in ('--dataid'):
                Options.dataid = v
            elif o in ('--online'):
                Options.env = 'online'
            elif o in ('--offline'):
                Options.env = 'offline'
            elif o in ('-D', '--degradation'):
                Options.degradation = True
            elif o in ('--show'):
                Options.show_win_file = v
            elif o in ('--http'):
                Options.http = True
            elif o in ('--start'):
                Options.daemon_action = 'start'
            elif o in ('--stop'):
                Options.daemon_action = 'stop'
            elif o in ('--restart'):
                Options.daemon_action = 'restart'
            elif o in ('--daemon'):
                Options.daemon = True
            elif o in ('--http-port'):
                Options.http_port = int(v)
            elif o in ('--tenant'):
                Options.tenant = v
            elif o in ('--tenant_id'):
                Options.tenant_id = int(v)
            else:
                assert False, 'unhandled option [%s]' & o
        return args

    def __ignore_signal(self):
        def signal_handler(signal, frame):
            pass
        signal.signal(signal.SIGINT, signal_handler)

    def __myprint(self, info):
        print info
        print
        self.__usage()

    def __init__(self):
        global oceanbase
        self.__set_env()

        if "print-config" in self.__parse_options():
            print "".join(ObConfigHelper().get_config(Options.dataid))
            ObConfigHelper().get_dataid_list()
            exit(0)
        if not Options.show_win_file:
            oceanbase = OceanBase(Options.dataid)
            if Options.using_ip_port:
                oceanbase.test_alive(do_false=self.__myprint)

        if Options.http:
            pid_file = '/tmp/dooba.%d.pid' % Options.http_port
            if Options.daemon:
                if Options.daemon_action == 'start':
                    HTTPServerDaemon(pid_file).start()
                elif Options.daemon_action == 'stop':
                    HTTPServerDaemon(pid_file).stop()
                elif Options.daemon_action == 'restart':
                    HTTPServerDaemon(pid_file).restart()
            else:
                try:
                    HTTPServerDaemon(pid_file).run()
                except KeyboardInterrupt:
                    pass
        else:
            try:
                curses.wrapper(Dooba)
            except KeyboardInterrupt:
                pass


class HTTPServerDaemon(Daemon):
    def run(self):
        #oceanbase.update_cluster_info()
        oceanbase.update_tenant_info()
        if (Options.tenant_id is not None):
            oceanbase.switch_tenant(Options.tenant_id)
        elif (Options.tenant is not None):
            oceanbase.switch_tenant(Options.tenant)
        oceanbase.start()
        server = BaseHTTPServer.HTTPServer(('', Options.http_port), WebRequestHandler)
        server.serve_forever()


class JSONDateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        elif isinstance(obj, (timedelta)):
            return 'NULL'
        else:
            return json.JSONEncoder.default(self, obj)


class WebRequestHandler(BaseHTTPServer.BaseHTTPRequestHandler):
    def do_GET(self):
        self.wfile.write('HTTP/1.1 200 OK\nContent-Type: text/html\n\n')
        result = {}
        latest = oceanbase.now()
        for svr in oceanbase.get_tenant_svr()['observer']:
            ip = svr['ip']
            port = svr['port']
            ipport = ip + ":" + port
            f = ColumnFactory('observer', ipport)
            columns = SQLPage.generate_columns(f)
            result["%s:%s"%(ip,port)] = {}
            for c in columns:
                v = c.value(latest)
                if type(v) == str:
                    v = v.strip()
                result["%s:%s"%(ip,port)][c.name()] = v
        self.wfile.write(json.dumps(result, cls=JSONDateTimeEncoder))

def main():
    DoobaMain()

if __name__ == "__main__":
    DoobaMain()
#
# dooba ends here