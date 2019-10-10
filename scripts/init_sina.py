import argparse
import os
import uuid

parser = argparse.ArgumentParser()
parser.add_argument("--user", default="cdoutrix", help="username to connect to cassandra")
parser.add_argument("--token", default=None, help="token to log, will try to read from ${HOME}/.cassandra/cqlshrc")
parser.add_argument("--keyspace", default=None, help="keyspace, will try to read from ${HOME}/.cassandra/cqlshrc")
parser.add_argument("--cluster", default="192.168.64.8", help="Cluster")
parser.add_argument("--sina", help="type of sina datastore", default="sql", choices=["sql", "cass"])
parser.add_argument("--sina_db", help="type of sina datastore", default="sina.sql", choices=["sql", "cass"])
args = parser.parse_args()

if args.sina == "sql":
    import sina.datastores.sql as sina
    if os.path.exists(args.sina_db):
        os.remove(args.sina_db)
    factory = sina.DAOFactory(db_path=args.sina_db)
else:
    import sina.datastores.cass as sina
    factory = sina.DAOFactory(keyspace=args.keyspace, node_ip_list=[args.cluster,])

from sina.model import Record, generate_record_from_json
record_handler = factory.create_record_dao()

# Create users
user = Record(id=uuid.uuid1().hex, type="user")
user.add_data("username", "cdoutrix")

record_handler.insert(user)

print("recs users:", list(record_handler.get_all_of_type("user")))