from  kosh import KoshStore
store = KoshStore(engine="cassandra", username="cdoutrix", token="OcJSDIFN2eTjg10cy/Lu/zRK8y8Jx6lmcZtxM4baat0=", cluster=["localhost",], keyspace="cdoutrix_k")

datasets = store.search("whynot", publisher="Some pub", whynot=6.7)

#for i, ds in enumerate(datasets):
#    print(i,ds)
print(datasets[0])