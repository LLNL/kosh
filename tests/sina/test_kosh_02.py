from  kosh import KoshStore
from kosh.sina import KoshSinaFile
from sina.utils import DataRange

store = KoshStore(engine="sina", username="cdoutrix", sql='sql', db_path='sina.sql')
DS = store.create(metadata={"publisher":"Some pub", "attr1":6, "whynot":6.7})


DS.add_file("sina.sql", "text")

print(DS)


#print(DS.__associated_data__)

myfile = DS.loadFromStore(DS.__associated_data__[0])
print(type(myfile))

myother = KoshSinaFile(Id=DS.__associated_data__[0], mimetype="text", store=DS.__store__)
print(type(myother))


search = store.search(publisher=DataRange("", "ZZZZ"), attr1=6)
print(len(search), search[0])

search = store.search(publisher=DataRange("", "ZZZZ"), attr1=6, ids_only=True)
print(len(search), search[0])