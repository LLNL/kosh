# Cassandra implementation
from kosh.core import KoshStoreBaseClass,  KoshDatasetBaseClass, KoshArrayBaseClass
import cassandra
from cassandra.cluster import Cluster, Session
from cassandra.auth import PlainTextAuthProvider
from cassandra.cqlengine.models import Model
from cassandra.cqlengine import connection, columns
from cassandra.cqlengine.management import sync_table
from cassandra.util import uuid_from_time
from cassandra.query import BatchStatement, ConsistencyLevel
from .models import DataSetModel, UsersModel, MetadataModel, types_mapping
from collections import OrderedDict
import warnings
import time
import uuid
import numpy


types = {"dataset": 0, "array": 1, "image": 2}


class KoshBaseCassandraObject(object):
    def __init__(self, Id, koshType, protected=[]):
        self.__dict__["__protected__"] = [
            "__id__", "__type__", "__protected__"] + protected
        self.__dict__["__id__"] = Id
        self.__dict__["__type__"] = koshType

    def __setattr__(self, name, value):
        if name in self.__protected__:
            self.__dict__[name] = value
            return
        exists = MetadataModel.objects(
            id=self.__id__, id_type=self.__type__, name=name)
        if exists.count != 0:
            # we need to drop it first, replacing
            for e in exists:
                e.delete()
        MetadataModel.create(
            id=self.__id__, id_type=self.__type__, name=name, value=repr(value))

    def __getattr__(self, name):
        if name in self.__dict__["__protected__"]:
            return self.__dict__.get(name)
        if name == "__attributes__":
            return self.__getattributes__()

        exists = MetadataModel.objects(
            id=self.__id__, id_type=self.__type__, name=name)
        if exists.count() == 0:
            raise RuntimeError(
                "Object {} does not have {} attribute".format(self.__id__, name))
        return eval(exists[0].value)

    def __delattr__(self, name):
        if name in self.__protected__:
            return
        exists = MetadataModel.objects(
            id=self.__id__, id_type=self.__type__, name=name)
        if exists.count() == 0:
            raise RuntimeError(
                "Object {} does not have {} attribute".format(self.__id__, name))
        exists[0].delete()

    def listattributes(self):
        exists = MetadataModel.objects(id=self.__id__, id_type=self.__type__)
        attributes = []
        for e in exists:
            attributes.append(e.name)
        return attributes

    def __getattributes__(self):
        exists = MetadataModel.objects(id=self.__id__, id_type=self.__type__)
        attributes = {}
        for e in exists:
            attributes[e.name] = eval(e.value)
        return attributes

class KoshConnectBase(object):
    def __init__(self, username, token, keyspace=None, cluster=["sonar8", ], auth_provider=None, cassandraRoot="kosh"):
        # This allows me to have test tables, etc...
        self.__cassandraRoot__=cassandraRoot
        print("Connecting")
        self.connect(username, token, keyspace, cluster, auth_provider)

    def connect(self, username, token, keyspace=None, cluster_names=["sonar8", ], auth_provider=None):
        """ Connect to a cassandra database """
        from cassandra.cluster import Cluster, Session
        if auth_provider is None:
            print("CREATING AUTH", username, token)
            from cassandra.auth import PlainTextAuthProvider
            auth_provider=PlainTextAuthProvider(
                username=username, password=token)

        if not isinstance(cluster_names, list):
            cluster_names = [cluster_names,]
        # print("CLUSTER TO:", cluster_names)
        cluster=Cluster(cluster_names, auth_provider=auth_provider)
        if keyspace is None:
            keyspace=username+"_k"
        # print("KSPACE:", keyspace)
        self.__session__=cluster.connect(keyspace)
        # print("CONNECTED TO:", self.__session__, auth_provider.username, auth_provider.password)
        self.__username__=username
        connection.setup(cluster_names, default_keyspace=keyspace,
                         auth_provider=auth_provider)
        for obj in [DataSetModel, UsersModel, MetadataModel]:
            obj.__keyspace__=self.__session__.keyspace
            obj.__table_name__="{}_{}".format(
                self.__cassandraRoot__, obj.__table_suffix__)
            sync_table(obj)
        self.__user_id__=UsersModel.objects(name=self.__username__)[0].id
        # self.prepared = {}
        # self.prepared["datasetids"] = self.session.prepare("SELECT id from {}_datasets".format(self.cassandraRoot))
        # self.prepared["metadata"] = self.session.prepare("SELECT * from {}_metadata where id=? and id_type=?".format(self.cassandraRoot))

    def __del__(self):
        # when destroying let's shutdown Cassandra cluster
        # print("Shutting down Cassandra Cluster")
        self.__session__.cluster.shutdown()
        del(self.__session__)


class KoshArrayCassandra(KoshConnectBase, KoshBaseCassandraObject, KoshArrayBaseClass):
    def __init__(self, Id, dimensions, username, token, keyspace=None, cluster=["sonar8", ], auth_provider=None, cassandraRoot="kosh"):
        """ Create a Casandra-based array, needs a connection to a cassandra database for metadata

        Id: can be set to None to indicate new/inexisting array, otherwise point to array to read/extend
        dimensions: ignored if array already exists, otherwise dictionary with dimension names as keys and metadata as dictinoary value
        """
        # print("IN INIT OF KoshArrayCassandra")
        if Id is None:  # New array?
            Id = uuid.uuid1().hex

        KoshBaseCassandraObject.__init__(self, Id, types["array"], protected=[
                                          "__name__", "__creator__", "__type__", "__dims__",
                                          "__session__", "__username__", "__cassandraRoot__",
                                          "__table_id__", "__user_id__",
                                          "__readers__", "__exporters__"])

        KoshConnectBase.__init__(self, username, token, keyspace, cluster, auth_provider, cassandraRoot)

        self.__type__ = types["array"]
        # Ok now create associted table, based on dimensions
        self.__table_id__ = "{}_array_{}".format(self.__cassandraRoot__, Id)

        # class ArrayModel(Model):
        #     value = columns.Float(primary_key=True)
        #     __table_name__ = self.__table_id__

        dims = []
        primary = []
        if isinstance(dimensions, (list, tuple)):
            tmp = OrderedDict()
            for d in dimensions:
                tmp[d] = {}
            dimensions = tmp
        if not isinstance(dimensions, OrderedDict):
            raise RuntimeError("'dimensions' must be list or ordered dict")
        self.__dims__ = dimensions
        for dim in dimensions:
            dim_dict = dimensions[dim]
            dim_type = dim_dict.get("type", "float")
            # kargs = {}
            if dim_dict.get("primary", False):
                primary.append(dim)
            #     kargs["primary_key"] = True
            #  setattr(ArrayModel, dim, types_mapping[dim_type](**kargs))
            dims += [dim+" "+dim_type]
        if primary == []:
            primary=dimensions.keys()
        tables=self.__session__.cluster.metadata.keyspaces[self.__session__.keyspace].tables
        dims.append("value float")
        if not self.__table_id__ in tables:
            self.__session__.execute("create table {}({}, primary key({}))".format(
                self.__table_id__, ",".join(dims), ",".join(primary)))
        # self.__ArrayModel__ = ArrayModel

    def load_from_numpy(self, data, offsets = None):
        dims = list(self.__dims__.keys()) + ["value", ]
        vals = ["?",] * len(dims)
        ps = self.__session__.prepare("INSERT INTO {} ( {} ) VALUES ( {} )".format(self.__table_id__, ",".join(dims), ",".join(vals)))
        # Ok did user send us dimensions values?
        for i, d in enumerate(self.__dims__):
            if self.__dims__[d].get("values", None) is None:
                self.__dims__[d]["values"] = list(range(data.shape[i]))
        
        batch = BatchStatement(consistency_level=ConsistencyLevel.QUORUM)
        for i in range(data.size):
            values = list(numpy.unravel_index(i, data.shape))
            values += [float(data.flat[i]),]
            batch.add(ps, values)
        self.__session__.execute_async(batch)

    def __getitem__(self, *args):
        args= args[0]
        dims = list(self.__dims__.keys())
        values = []
        for i in range(len(args)):
            a = self.__dims__[dims[i]]["values"][args[i]]
            vals = ", ".join([str(_) for _ in a])
            cmd = dims[i]+" IN ( {} )".format(vals)
            values.append(cmd)
        stmnt = "SELECT * FROM "+self.__table_id__ + " WHERE "+ " AND ".join(values)+";"
        return self.__session__.execute(stmnt)
    




    def __str__(self):
        st=""
        st += "KOSH Array\n"
        st += "\tid: {}\n".format(self.__id__)
        st += "\tdimensions:{}\n".format(self.__dims__)
        atts=self.__attributes__
        if len(atts) > 0:
            st += "\n--- Attributes ---\n"
            for a in atts:
                st += "\t{}: {}\n".format(a, atts[a])
        st += "DONE!"
        return st

class KoshDatasetCassandra(KoshDatasetBaseClass, KoshBaseCassandraObject):
    def __init__(self, Id):
        KoshBaseCassandraObject.__init__(self, Id, types["dataset"], protected=[
                                          "__name__", "__creator__"])
        ds=DataSetModel.objects(id=Id)[0]  # unique by design
        self.__creator__=ds.creator
        self.__name__=ds.name

    def __str__(self):
        st=""
        st += "KOSH DATASET\n"
        st += "\tid: {}\n".format(self.__id__)
        st += "\tname:{}\n".format(self.__name__)
        st += "\tcreator: {}\n".format(self.__creator__)
        atts=self.__attributes__
        if len(atts) > 0:
            st += "\n--- Attributes ---\n"
            for a in atts:
                st += "\t{}: {}\n".format(a, atts[a])
        return st



class KoshStoreCassandra(KoshConnectBase, KoshStoreBaseClass):
    def search(self, *atts, **keys):
        """ Search cassandra for datasets matching some metadata
        arguments are the metadata name we are looking for e.g search("attr1", "attr2") 
        default is to AND, but can be changed via the __cassandra_search_operator keyword
        """
        print("ARGS:", atts)
        print("KARGS:", keys)
        if "__cassandra_search_operator" in keys:
            __cassandra_search_operator = keys.pop("__cassandra_search_operator")
        else:
            __cassandra_search_operator = "AND"
        no_values = __cassandra_search_operator.join([ "name = '{}'".format(k) for k in atts]) 
        values = __cassandra_search_operator.join(["( name = '{}' AND value = '{}' )".format(k,v) for k,v in keys.items()])
        if no_values != "" and values != "":
            search_terms = no_values + " AND " + values
        elif values == "":
            search_terms = values
        else:
            search_terms = no_values

        if search_terms == "":
            raise RuntimeError("You need to pass some search argument")

        search_params = "id_type={} AND ( {} )".format(types["dataset"], search_terms)
        print("SEARCH :", search_params)
        rows = self.__session__.execute("select id from {}_metadata where {}".format(self.__cassandraRoot__, search_params))
        for row in rows:
            print(row.id)
    def open(self, datasetId):
        #warnings.warn("Not implemented yet")
        return KoshDatasetCassandra(datasetId)

    def create(self, name=None, datasetId=None, metadata={}):
        """create a new (possibly named) dataset"""
        # existingDatasets = self.session.execute_async(self.prepared["datasetids"])
        if datasetId is None:
            if name is None:
                name="Unnamed Dataset"
            print("CREATING NEW DS")
            ds=DataSetModel.create(id=uuid_from_time(time.time()), name=name, creator=self.__user_id__)
        else:
            ds=DataSetModel.objects(id=datasetId)
            if ds.count() == 0:
                ds=DataSetModel.create(
                    id=datasetId, name=name, creator=self.__user_id__)
            else:
                raise RuntimeError(
                    "Dataset Id {}, already exists, cannot create duplicate dataset".format(datasetId))
        ds=KoshDatasetCassandra(str(ds.id))
        for name in metadata:
            setattr(ds, name, metadata[name])
        return ds
