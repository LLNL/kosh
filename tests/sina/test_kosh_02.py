from  kosh import KoshStore
from kosh.sina import KoshSinaFile
from sina.utils import DataRange

store = KoshStore(engine="sina", username="cdoutrix", sql='sql', db_path='sina.sql')
DS = store.create(metadata={"publisher":"Some pub", "attr1":6, "whynot":6.7})


DS.add_file("sina.sql", "text")

print(DS)


#print(DS.associated_data)

myfile = DS.loadFromStore(DS.associated_data[0])
print(type(myfile))

myother = KoshSinaFile(Id=DS.associated_data[0], filetype="text", store=DS.__store__)
print(type(myother))


search = list(store.search(publisher=DataRange("", "ZZZZ"), attr1=6))

print(len(search))

