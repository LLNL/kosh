
from  kosh import KoshStore
from kosh.sina import KoshSinaFile
from sina.utils import DataRange

store = KoshStore(engine="sina", username="cdoutrix", sql='sql', db_path='sina.sql')
DS = store.create(metadata={"publisher":"Some pub", "attr1":6, "whynot":6.7})


DS.add_file("/p/lscratchh/cdoutrix/cdoutrix/IBM/workaround/base/molar.0.3_shock.1.1_taper.0.8_skew.0.8/IBM/extracts", "kull")

print(DS)

kull = DS.open(DS.__associated_data__[0])
print("KULL AXES:", kull.getAxisList("node"))