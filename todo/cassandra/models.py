from cassandra.cqlengine.models import Model
from cassandra.cqlengine import columns
import time

types_mapping = {
    "int": columns.Integer,
    "text": columns.Text,
    "uuid": columns.TimeUUID,
    "float": columns.Float,
}


class DataSetModel(Model):
    __table_suffix__ = "datasets"
    id = columns.TimeUUID(primary_key=True)
    creator = columns.Integer(required=True, primary_key=True)
    name = columns.Text()


class UsersModel(Model):
    __table_suffix__ = "users"
    name = columns.Text(primary_key=True, required=True)
    id = columns.Integer(primary_key=True)


class MetadataModel(Model):
    __table_suffix__ = "metadata"
    id = columns.TimeUUID(primary_key=True)

    name = columns.Text(primary_key=True, required=True)
    id_type = columns.Integer(primary_key=True)
    value = columns.Text(primary_key=True)
