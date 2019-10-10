import kosh
from kosh.cassandra import KoshArrayCassandra
import numpy
cluster = "localhost"
dimensions = ["a", "b", "c"]
a = KoshArrayCassandra(Id=None, dimensions=dimensions,username="cdoutrix", token="OcJSDIFN2eTjg10cy/Lu/zRK8y8Jx6lmcZtxM4baat0=", cluster=cluster, keyspace="cdoutrix_k")
a.name = "Charles D."
print(a)

def row_to_numpy(rows):
    data = []
    for row in rows:
        data.append(row.value)
    return numpy.array(data)

data = numpy.arange(3*5*6)
data.shape= (3,5,6)
a.load_from_numpy(data)
print("A:", a)
sub_data = row_to_numpy(a[1:3, 2:4])
print(sub_data)
sub_orig = data[1:3, 2:4].flat[:]
print(sub_data - sub_orig)