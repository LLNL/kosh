import os
import random
from koshbase import KoshTest


class KoshTestEnsembleCP(KoshTest):

    def test_ens_cp(self):
        s1, s1nm = self.connect()
        s2, s2nm = self.connect()
        s3, s3nm = self.connect()
        s4, s4nm = self.connect()

        e1 = s1.create_ensemble()
        d1 = e1.create()
        name = f"kosh_{random.randint(0, 23434434)}.txt"
        name_a = f"kosh_ensemble_test_{random.randint(0, 23434434)}.txt"
        name_b = f"kosh_ensemble_test_{random.randint(0, 23434434)}.txt"
        for nm in [name, name_a, name_b]:
            with open(nm, "w") as f:
                print("test", file=f)
            d1.associate(nm, "txt")

        # --------------STORE1----------
        for datastore in s1.find():
            print(datastore)

        # Copying files
        name2 = f"kosh_{random.randint(0, 23434434)}.txt"
        s1.cp(name, name2, destination_stores=s2)
        for datastore in s1.find():
            print(datastore)

        # Confirming files in Datastore 1 after mv/cp
        d1 = next(s1.find())
        a1 = d1.find(mime_type="txt")
        list_of_files = [os.path.basename(x.uri) for x in a1]
        self.assertCountEqual(list_of_files, [name, name_a, name_b, name2])

        # Confirming ensemble ID in Datastore 2 after mv/cp
        e2 = list(s2.find_ensembles())[0]
        self.assertEqual(e2.id, e1.id)

        # Confirming datastore ID in Datastore 2 after mv/cp
        e2 = s2.find_ensembles()
        d2 = list(list(e2)[0].find_datasets())[0]
        self.assertEqual(d2.id, d1.id)

        # Confirming files in Datastore 2 after mv/cp
        e2 = s2.find_ensembles()
        d2 = list(list(e2)[0].find_datasets())
        a2 = list(d2[0].find(mime_type="txt"))
        list_of_files = [os.path.basename(x.uri) for x in a2]
        self.assertCountEqual(list_of_files, [name, name_a, name_b, name2])

        # --------------STORE2----------
        for datastore in s2.find():
            print(datastore)

        # Copying files
        name3 = f"kosh_{random.randint(0, 23434434)}.txt"
        s2.cp(name2, name3, destination_stores=s3)
        for datastore in s2.find():
            print(datastore)

        # Confirming files in Datastore 2 after mv/cp
        d2 = next(s2.find())
        a2 = d2.find(mime_type="txt")
        list_of_files = [os.path.basename(x.uri) for x in a2]
        self.assertCountEqual(list_of_files, [name, name_a, name_b, name2, name3])

        # Confirming ensemble ID in Datastore 3 after mv/cp
        e3 = next(s3.find_ensembles())
        self.assertEqual(e3.id, e1.id)

        # Confirming datastore ID in Datastore 3 after mv/cp
        e3 = s3.find_ensembles()
        d3 = list(list(e3)[0].find_datasets())[0]
        self.assertEqual(d3.id, d1.id)

        # Confirming files in Datastore 3 after mv/cp
        e3 = s3.find_ensembles()
        d3 = list(list(e3)[0].find_datasets())
        a3 = list(d3[0].find(mime_type="txt"))
        list_of_files = [os.path.basename(x.uri) for x in a3]
        self.assertCountEqual(list_of_files, [name,  name_a, name_b, name2, name3])

        # --------------STORE3----------
        for datastore in s3.find():
            print(datastore)

        # Moving files
        name4 = f"kosh_{random.randint(0, 23434434)}.txt"
        s3.mv(name, name4, destination_stores=s4)
        for datastore in s3.find():
            print(datastore)

        # Confirming files in Datastore 3 after mv/cp
        d3 = next(s3.find())
        a3 = d3.find(mime_type="txt")
        list_of_files = [os.path.basename(x.uri) for x in a3]
        self.assertCountEqual(list_of_files, [name_a, name_b, name2, name3, name4])

        # Confirming ensemble ID in Datastore 4 after mv/cp
        e4 = next(s4.find_ensembles())
        self.assertEqual(e4.id, e1.id)

        # Confirming datastore ID in Datastore 4 after mv/cp
        e4 = s4.find_ensembles()
        d4 = list(list(e4)[0].find_datasets())[0]
        self.assertEqual(d4.id, d1.id)

        # Confirming files in Datastore 4 after mv/cp
        e4 = s4.find_ensembles()
        d4 = list(list(e4)[0].find_datasets())
        a4 = list(d4[0].find(mime_type="txt"))
        list_of_files = [os.path.basename(x.uri) for x in a4]
        self.assertCountEqual(list_of_files, [name_a, name_b, name2, name3, name4])

        # --------------STORE4----------
        for datastore in s4.find():
            print(datastore)

        # Moving Files
        name5 = f"kosh_{random.randint(0, 23434434)}.txt"
        s4.mv(name2, name5, destination_stores=s1, merge_strategy='overwrite')
        for datastore in s4.find():
            print(datastore)

        # Confirming files in Datastore 4 after mv/cp
        d4 = next(s4.find())
        a4 = d4.find(mime_type="txt")
        list_of_files = [os.path.basename(x.uri) for x in a4]
        self.assertCountEqual(list_of_files, [name_a, name_b, name3, name4, name5])

        # Confirming files in Datastore 1 after mv/cp
        d1 = next(s1.find())
        a1 = d1.find(mime_type="txt")
        list_of_files = [os.path.basename(x.uri) for x in a1]
        self.assertCountEqual(list_of_files, [name_a, name_b, name3, name4, name5])

        # os.remove(name)
        os.remove(name_a)
        os.remove(name_b)
        # os.remove(name2)
        os.remove(name3)
        os.remove(name4)
        os.remove(name5)
        s1.close()
        s2.close()
        s3.close()
        s3.close()
        os.remove(s1nm)
        os.remove(s2nm)
        os.remove(s3nm)
        os.remove(s4nm)
