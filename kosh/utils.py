from subprocess import Popen, PIPE
import os
import shlex
import sys


def create_new_db(name, engine='sina', db='sql', token="", keyspace=None):
    """create_new_db creates a new databasefor Kosh, adds a single user

    :param name: name of database
    :type name: str
    :param engine: engine to use, defaults to 'sina'
    :type engine: str, optional
    :param db: type of database for engine, defaults to 'sql', can be 'cass'
    :type db: str, optional
    :param token: for cassandra connection, token to use, defaults to "" means try to retrieve from user home dir
    :type token: str, optional
    :param keyspace: for cassandra keyspace to use, defaults to None means [user]_k
    :type keyspace: str, optional
    """
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
