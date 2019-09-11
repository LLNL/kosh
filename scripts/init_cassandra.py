#!/usr/bin/env python

import argparse
from cassandra.cluster import Cluster, Session
from cassandra.auth import PlainTextAuthProvider
import os

parser = argparse.ArgumentParser()
parser.add_argument("--user", default="cdoutrix", help="username to connect to cassandra")
parser.add_argument("--token", default=None, help="token to log, will try to read from ${HOME}/.cassandra/cqlshrc")
parser.add_argument("--keyspace", default=None, help="keyspace, will try to read from ${HOME}/.cassandra/cqlshrc")
parser.add_argument("--cluster", default="sonar8", help="Cluster")
parser.add_argument("--tables_root", default="dkosh", help="Cluster")
args = parser.parse_args()

token = args.token
keyspace = args.keyspace
with open(os.path.expanduser("~/.cassandra/cqlshrc")) as f:
    lines = f.readlines()
    if token is None:
        token = lines[2].split("=", 0)[-1].strip()
    if keyspace is None:
        keyspace = lines[3].split("=")[1].strip()
auth_provider = PlainTextAuthProvider(username=args.user, password=token)
cluster = Cluster(args.cluster.split(),auth_provider=auth_provider)#, protocol_version=2)
session = cluster.connect(keyspace)

drop_commands = """
drop table {root}_metadata
drop table {root}_datasets
drop table {root}_permissions
drop table {root}_targets
""".format(root=args.tables_root)

create_commands = """
create table {root}_metadata (id timeuuid , id_type int , name text, value text, primary key (id, id_type, name))
create table {root}_datasets (id timeuuid primary key, name text)
create table {root}_permissions(id timeuuid primary key, type int, user int, permission int)
create table {root}_targets(id timeuuid primary key, soure_type int, source_parameters map<text, text>, target_url text)
""".format(root=args.tables_root)
for command in drop_commands.split("\n"):
    if len(command)>0:
        print("Executing:", command)
        try:
            session.execute(command)
        except Exception:
            pass

# Now drop array tables
tables = cluster.metadata.keyspaces[keyspace].tables
for table in tables:
    if table.startswith("{}_array".format(args.tables_root)):
        try:
            session.execute("drop table {}".format(table))
        except Exception:
            pass

# Create tables
for command in create_commands.split("\n"):
    if len(command)>0:
        print("Executing:", command)
        session.execute(command)
