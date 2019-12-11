from subprocess import Popen, PIPE
import os
import shlex
import sys

def create_new_db(name, engine='sina', db='sql', token="", keyspace=None):
    user = os.environ["USER"]
    if db == 'sql' and name[-4:].lower() != ".sql":
        name += ".sql"
    if engine == "sina":
        cmd = "{}/init_sina.py --user={} --sina={} --sina_db={}".format(
            sys.prefix+"/bin",
            user,
            db,
            name)
    elif engine == 'cassandra':
        if keyspace is None:
            keyspace = user+"_k"
        cmd = "{}/init_cassandra.py --user={} --token={}" \
            "--keyspace={} --tables_root={} --cluster={}".format(
                sys.prefix+"/bin",
                user,
                token,
                keyspace,
                db)
    p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE)
    o, e = p.communicate()
