from  kosh import KoshStore
store = KoshStore(engine="cassandra", username="cdoutrix", token="OcJSDIFN2eTjg10cy/Lu/zRK8y8Jx6lmcZtxM4baat0=", cluster=["localhost",], keyspace="cdoutrix_k")
#DS = store.create()  # metadata={"publisher":"Some pub", "attr1":6, "whynot":6.7})
DS = store.create(metadata={"publisher":"Some pub", "attr1":6, "whynot":6.7})
rows = store.__session__.execute("select * from kosh_datasets")
for r in rows:
    print(r)


print(DS)
DS.charles = "Charles"
print(DS)
DS.charles = "Charles Doutriaux"
print(DS)
del(DS.charles)
print(DS)
try:
    print(DS.whynot)
except:
    print("failed printing inextisting attriubte as expected")
try:
    print(DS.charles)
except:
    print("Yep not there")
