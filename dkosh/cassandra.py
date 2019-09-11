# Cassandra implementation
from .core import DKoshStoreBaseClass,  DKoshDatasetBaseClass
import cassandra
from cassandra.cluster import Cluster, Session
from cassandra.auth import PlainTextAuthProvider
from cassandra.cqlengine.models import Model
from cassandra.cqlengine import columns, connection
from cassandra.cqlengine.management import sync_table

import warnings
import time

class DataSetModel(Model):
    id = columns.TimeUUID(primary_key=True,
                          default=cassandra.util.uuid_from_time(time.time()))
    name = columns.Text(required=True, default='')

class DKoshDatasetCassandra(DKoshDatasetBaseClass):
    def __init__(self, datasetId, session):
        self.datasetId = datasetId
        self.session = session

class DKoshStoreCassandra(DKoshStoreBaseClass):
    def __init__(self, username, token, keyspace=None, cluster=["sonar8",], auth_provider=None, cassandraRoot="dkosh"):
        # This allows me to have test tables, etc...
        self.cassandraRoot =  cassandraRoot
        self.types = {"dataset": 0}
        self.connect(username, token, keyspace, cluster, auth_provider)

    def connect(self, username, token, keyspace=None, cluster_names=["sonar8",], auth_provider=None):
        """ Connect to a cassandra database """
        from cassandra.cluster import Cluster, Session
        if auth_provider is None:
            from cassandra.auth import PlainTextAuthProvider
            authProvider = PlainTextAuthProvider(username=username, password=token)
        cluster = Cluster(cluster_names, auth_provider=auth_provider)
        if keyspace is None:
            keyspace = username+"_k"
        self.session = cluster.connect(keyspace)
        connection.setup(cluster_names, default_keyspace=keyspace, auth_provider=auth_provider)
        DataSetModel.__table_name__ = "{}_datasets".format(self.cassandraRoot)
        DataSetModel.__keyspace__ = self.session.keyspace
        sync_table(DataSetModel)
        self.prepared = {}
        self.prepared["datasetids"] = self.session.prepare("SELECT id from {}_datasets".format(self.cassandraRoot))
        self.prepared["metadata"] = self.session.prepare("SELECT * from {}_metadata where id=? and id_type=?".format(self.cassandraRoot))
    def search(self, keys):
        warnings.warn("Not implemented yet")
        return []

    def open(self, datasetId):
        warnings.warn("Not implemented yet")
        return DKoshDatasetCassandra(datasetId)

    def create(self, name=None, datasetId=None, metadata={}):
        """create a new (possibly named) dataset"""
        existingDatasets = self.session.execute_async(self.prepared["datasetids"])
        if datasetId is None:
            if name is None:
                name = "Unnamed Dataset"
            DataSetModel.create(name=name)



