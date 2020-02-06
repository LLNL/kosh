from  kosh import KoshStore
store = KoshStore(engine="sina", username="cdoutrix", sql='sql', db_path='sina.sql')
DS = store.create(metadata={"publisher":"Some pub", "attr1":6, "whynot":6.7})
DS = store.create(metadata={"publisher":"Some other pub", "attr1":16, "whynot":6.7})
DS = store.create(metadata={"publisher":"Some random pub", "attr1":62, "ofcourse":.7})
print(DS.__attributes__)

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
