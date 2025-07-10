from koshbase import KoshTest
import os


def find_json_files(directories):
    """
    Walk through a directory and its subdirectories to find all .json files.

    Args:
        directories (list/str): The root directories to start searching from.

    Returns:
        list: A list of file paths for all .json files found.
    """
    json_files = []

    if isinstance(directories, str):
        directories = [directories,]
    for directory in directories:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".json"):
                    json_files.append(os.path.join(root, file))

    return json_files


class KoshTestList(KoshTest):
    def test_cache_list_features(self):
        store, uri = self.connect()

        for filename in find_json_files([os.getcwd(), '../sina', 'sina']):
            print("file:", filename)
            if "kosh_test_venv" in filename or "tests/kosh_export.json" in filename:  # from other test
                continue
            store.import_dataset(filename)
            store.delete_all_contents(force="SKIP PROMPT")
        store.close()
        os.remove(uri)


if __name__ == "__main__":
    A = KoshTestList()
    for nm in dir(A):
        if nm.startswith("test_"):
            fn = getattr(A, nm)
            print(nm, fn)
            fn()
