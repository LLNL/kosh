#!/usr/bin/env python

import argparse
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--user",
    default="cdoutrix",
    help="username to connect to cassandra")
parser.add_argument(
    "--token",
    default=None,
    help="token to log, will try to read from ${HOME}/.cassandra/cqlshrc")
parser.add_argument(
    "--keyspace",
    default=None,
    help="keyspace, will try to read from ${HOME}/.cassandra/cqlshrc")
parser.add_argument("--cluster", default="sonar8", help="Cluster")
parser.add_argument(
    "--tables_root",
    default="kosh",
    help="root for tables names")
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
# , protocol_version=2)
cluster = Cluster(args.cluster.split(), auth_provider=auth_provider)
session = cluster.connect(keyspace)

drop_commands = """
drop table {root}_metadata
drop table {root}_datasets
drop table {root}_permissions
drop table {root}_users
drop table {root}_targets
""".format(root=args.tables_root)

create_commands = """
create table {root}_metadata (id timeuuid , id_type int , name text, value text, primary key ((id), name, id_type, value))
create index on {root}_metadata (name)
create index on {root}_metadata (value)
create index on {root}_metadata (id_type)
create table {root}_datasets (id timeuuid, creator int, name text, primary key (id, creator))
create table {root}_users(id int, name text, primary key (name, id))
create table {root}_permissions(user_id int, resource_id text, resource_type int, permission int, primary key (user_id, resource_id, resource_type))
create table {root}_targets(id timeuuid primary key, source_type int, source_parameters map<text, text>, target_url text)
insert into {root}_users (id, name) values (0, '{user}')
""".format(root=args.tables_root, user=args.user)  # noqa
for command in drop_commands.split("\n"):
    if len(command) > 0:
        print("Executing:", command)
        try:
            session.execute(command)
        except Exception:
            pass

# Now drop array tables
tables = list(cluster.metadata.keyspaces[keyspace].tables.keys())
for table in tables:
    if table.startswith("{}_array".format(args.tables_root)):
        try:
            session.execute("drop table {}".format(table))
        except Exception as err:
            print("EEROR: ", err)
            pass

# Create tables
for command in create_commands.split("\n"):
    if len(command) > 0:
        print("Executing:", command)
        session.execute(command)
