from  kosh import KoshStore
store = KoshStore(engine="cassandra", username="cdoutrix", token="OcJSDIFN2eTjg10cy/Lu/zRK8y8Jx6lmcZtxM4baat0=", cluster=["localhost",], keyspace="cdoutrix_k")

store.search("whynot", publisher="Some Pub")