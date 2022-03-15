import koshbase
import os
import grp


class TestUsersGroups(koshbase.KoshTest):
    def test_add_user(self):
        store, uri = self.connect()

        with self.assertRaises(ValueError):
            store.add_user(os.environ.get("USER", "default"))

        store.add_user("kosh_test")
        self.assertEqual(len((list(store.find(kosh_type=store._users_type, ids_only=True)))), 3)
        with self.assertRaises(ValueError):
            store.add_user("kosh_test")
        os.remove(uri)

    def test_groups(self):
        store, uri = self.connect()
        unix_groups = [g[0] for g in grp.getgrall()]

        with self.assertRaises(ValueError):
            store.add_group(unix_groups[0])

        grp_name = "test_kosh"
        while grp_name in unix_groups:
            grp_name += "_blah"
        store.add_group(grp_name)
        self.assertEqual(len((list(store.find(types=store._groups_type, ids_only=True)))), 1)

        store.add_user("kosh_test_user", groups=[grp_name])
        self.assertEqual(len((list(store.find(types=store._users_type, ids_only=True)))), 3)

        os.remove(uri)


if __name__ == "__main__":
    A = TestUsersGroups()
    for nm in dir(A):
        if nm[:4] == "test":
            fn = getattr(A, nm)
            print(nm, fn)
            fn()
