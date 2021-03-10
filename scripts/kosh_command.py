#!/bin/sh sbang
#!/usr/bin/env python
# This implements Kosh's CLI
from __future__ import print_function
import argparse
import kosh
import sys
from sina.utils import DataRange
import shlex
from subprocess import Popen, PIPE
import os
import tempfile
import ast
import glob
import json
try:
    basestring
except NameError:
    basestring = str


def get_all_files(opts):
    """given a list of files, directory or pattern
    returns the list of files describe by these
    :param opts: list of files, directory or pattern
    :type opts: list
    :return: list of files
    :rtype: list
    """
    files = []
    for filename in opts:
        if os.path.isdir(filename):
            for root, dirs, filenames in os.walk(filename):
                for fnm in filenames:
                    files.append(os.path.join(root, fnm))
        elif "*" in filename or "?" in filename:
            files += glob.glob(filename)
        elif os.path.exists(filename):
            files.append(filename)
    return files


def find_files_from_list(uris):
    """given a uri/path walks this path/pattern to get sub dir/files
    :param uris: list of uris to scan (or list of lists)
    :type uris: list
    :return: List of files
    :rtype: list
    """
    # figure out targets
    new_uris = []
    for uri in uris:
        if isinstance(uri, list):
            new_uris += uri
        else:
            new_uris.append(uri)
    # figure out patterns and dirs
    pop = []
    for index, filename in enumerate(list(new_uris)):
        if "*" in filename or "?" in filename:
            # It's a pattern
            pop.append(index)
            new_uris += glob.glob(filename)
        elif os.path.isdir(filename):
            pop.append(index)
            for root, dirs, filenames in os.walk(filename):
                for fnm in filenames:
                    new_uris.append(os.path.join(root, fnm))

    # cleanup list
    for index in pop[::-1]:
        new_uris.pop(index)

    return new_uris


def core_parser(description,
                usage=None, prog=None):
    """
    Return the core parser with arguments common to all operations
    :param description: Description for the argparse parser
    :type description: str
    :param usage: Usage string
    :type usage: str
    :param prog: Name of the program for argparse to print
    :type prog: str
    :return: arparse parser
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=description,
        prog=prog,
        usage=usage,
        epilog="Kosh version {kosh.__version__}".format(kosh=kosh))
    parser.add_argument("--store", "-s", required=True,
                        help="Kosh store to use")
    parser.add_argument("--dataset_record_type", "-d", default="dataset",
                        help="type used by sina db that Kosh will recognize as dataset")
    parser.add_argument("--version", "-v", action="store_true",
                        help="print version and exit")
    return parser


def parse_metadata(terms):
    """
    Parse metadata for Kosh search
    param=value / param>value, etc...
    :param terms: list of strings conatining name/operator/value
    :term terms: list of str
    :return: Dictionary with name as key and matching sina search object as value
    :rtype: dict
    """
    metadata = {}
    for term in terms:
        found = False
        for operator in ["<=", ">=", "<", ">", "="]:
            sp = term.split(operator)
            if len(sp) != 2:
                continue
            found = True
            key = sp[0]
            try:
                # converts to int/float/DataRange if possible
                value = eval(sp[1])
            except Exception:
                value = sp[1]
            if operator == ">=":
                value = DataRange(min=value, min_inclusive=True)
            elif operator == ">":
                value = DataRange(min=value, min_inclusive=False)
            elif operator == "<=":
                value = DataRange(max=value, max_inclusive=True)
            elif operator == "<":
                value = DataRange(max=value, max_inclusive=False)
            metadata[key] = value
            break

        if not found:
            raise ValueError("Metadata must be in form 'key=value'")
    return metadata


def process_cmd(command, use_shell=False, shell="/usr/bin/bash"):
    """ Convenience function to run a command
    :param command: command to run
    :type command: str
    :param use_shell: ssh needs to be run as 'shell' and communicated command
                      This let us decide if it's our way to run the command
    :type use_shell: bool
    :param shell: If using a shell, this tells which shell to use
    :type shell: str
    :return: process object and output and error streams
    :rtype: list
    """

    if use_shell:
        proc = Popen(shell, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        o, e = proc.communicate(command.encode())
    else:
        proc = Popen(shlex.split(command), stdout=PIPE, stderr=PIPE)
        o, e = proc.communicate()
    return proc, o, e


def open_stores(uris, dataset_record_type):
    """Given a list of paths/uri to Kosh stores this opens them all
    :param uris: list of Kosh stores to open
    :type uris: list
    :param dataset_record_type: record type in the store(s) or list of
    :type dataset_record_type: list or str
    :return: list of kosh stores
    :rtype: list
    """
    stores = []
    if isinstance(dataset_record_type, basestring):
        dataset_record_type = [dataset_record_type, ] * len(uris)
    for index, store_uri in enumerate(uris):
        if "@" in store_uri:
            # Remote store let's go fetch it
            _, store_local_uri = tempfile.mkstemp()
            cmd = "scp {} {}".format(store_uri, store_local_uri)
            p, o, e = process_cmd(cmd, use_shell=True)
            if p.returncode != 0:
                raise RuntimeError("Could not fetch remote store:", store_uri)
        else:
            store_local_uri = store_uri
        store = kosh.KoshStore(
            db_uri=store_local_uri,
            dataset_record_type=dataset_record_type[index])
        stores.append(store)
    return stores


def close_stores(stores, uris):
    """Closes a list of stores and if it was remote send it back to remote
    :param stores: List of Kosh store objects
    :type stores: list
    :param uris: list of Kosh stores to open
    :type uris: list
    """
    for i, store in enumerate(stores):
        # store.close()
        if store.db_uri != uris[i]:
            # Ok this one is a temporary file we need to send it back to remote
            cmd = "scp {} {}".format(store.db_uri, uris[i])
            p, o, e = process_cmd(cmd, use_shell=True)
            if p.returncode != 0:
                raise RuntimeError(
                    "Could not send updated store to {}".format(
                        uris[i]))


class KoshCmd(object):
    """Engine to dispatch kosh command to appropriate function"""
    def __init__(self):
        commands = "".join(
            ["" if k[0] == "_" else "\n\t" + k for k in sorted(dir(self))])
        parser = core_parser(
            description='Execute kosh operations',
            usage='''kosh <command> [<args>]

Available commands are:
    {commands}
'''.format(commands=commands))
        parser.add_argument('command', help='Subcommand to run')
        # first we parse all of it to catch --version
        help = False
        if (
                "--help" in sys.argv or "-h" in sys.argv) and sys.argv[1] in commands:
            help = True
            # Let's temporarily yank help from sys.argv
            try:
                help_index = sys.argv.index("-h")
            except Exception:
                pass
            try:
                help_index = sys.argv.index("--help")
            except Exception:
                pass
            sys.argv.pop(help_index)
        args, _ = parser.parse_known_args(sys.argv + ["-s", "blah"])
        if args.version:
            print("Kosh version:", kosh.__version__)
            sys.exit(0)
        # Ok now we parse only the rest to catch the command
        # parse_args defaults to [1:] for args, but you need to
        # exclude the rest of the args too, or validation will fail
        args = parser.parse_args(sys.argv[1:2] + ["-s", "blah"])
        if not hasattr(self, args.command) or args.command[0] == "_":
            print('Unrecognized command: {args.command}'.format(args=args))
            print('Known commands: {}'.format(" ".join(commands)))
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        if help:
            sys.argv.insert(help_index, "--help")
        getattr(self, args.command)()

    def search(self):
        """search a store command"""
        parser = core_parser(
            description='Search Kosh store for datasets matching metadata in form key=value')
        parser.add_argument(
            "--print",
            "-p",
            help="print each dataset info",
            action="store_true")
        args, search_terms = parser.parse_known_args(sys.argv[2:])
        metadata = parse_metadata(search_terms)
        store = kosh.KoshStore(db_uri=args.store,
                               dataset_record_type=args.dataset_record_type)
        metadata["ids_only"] = True
        ids = store.search(**metadata)
        if args.print:
            for Id in ids:
                ds = store.open(Id)
                print(ds)
                print(
                    "=======================================================================")
        else:
            print("\n".join(ids))

    def cleanup_files(self):
        """Cleanup a store from references to dead files
        You can filter associated object by matching metadata in form key=value
        e.g mime_type=hdf5 will only dissociate non-existing files associated with mime_type hdf5
        some_att=some_val will only dissociate non-exisiting files associated and having the attribute "some_att" with value of "some_val"""
        parser = core_parser(
            prog="kosh clean",
            description="""Cleanup a store from references to dead files
        You can filter associated object by matching metadata in form key=value
        e.g mime_type=hdf5 will only dissociate non-existing files associated with mime_type hdf5
        some_att=some_val will only dissociate non-exisiting files associated and having the attribute
        'some_att' with value of 'some_val'""")
        parser.add_argument("--dry-run", "--rehearsal", "-r", "-D", help="Dry run only, list ids of dataset that would be cleaned up and path of files", action='store_true')
        parser.add_argument("--interactive", "-i", help="interactive mode, ask before dissociating", action="store_true")
        args, search_terms = parser.parse_known_args(sys.argv[2:])
        metadata = parse_metadata(search_terms)
        store = kosh.KoshStore(db_uri=args.store,
                               dataset_record_type=args.dataset_record_type)
        ids = store.search(ids_only=True)
        for Id in ids:
            ds = store.open(Id)
            missings = ds.cleanup_files(dry_run=args.dry_run, interactive=args.interactive, **metadata)
            if len(missings) != 0:
                if not args.interactive:  # already printed in interactive
                    print(ds)
                for uri in missings:
                    associated = ds.search(uri=uri)
                    if len(associated) != 0:
                        print("{} (mime_type={}) is missing".format(
                            associated[0].uri, associated[0].mime_type))

    def add(self):
        """add a dataset to a Kosh store command"""
        parser = core_parser(
            prog="kosh add",
            description='Adds a dataset to store')
        parser.add_argument(
            "--id", "-i", help="Desired Id for dataset", default=None)
        args, metadata = parser.parse_known_args(sys.argv[2:])
        metadata = parse_metadata(metadata)
        store = kosh.KoshStore(
            db_uri=args.store, dataset_record_type=args.dataset_record_type)
        ds = store.create(datasetId=args.id, metadata=metadata)
        print(ds.__id__)

    def remove(self):
        """Remove dataset(s) from store command"""
        parser = core_parser(
            prog="kosh remove",
            description='Removes a dataset from store')
        parser.add_argument("--ids", "-i", help="ids of datasets to print",
                            nargs="*", required=True, action="append")
        parser.add_argument("--force", "-f", action="store_true",
                            help="remove without asking for confirmation")
        args = parser.parse_args(sys.argv[2:])
        datasets = []
        for i in args.ids:
            datasets += i

        store = kosh.KoshStore(
            db_uri=args.store, dataset_record_type=args.dataset_record_type)
        for Id in datasets:
            if args.force:
                store.delete(Id)
            else:
                ds = store.open(Id)
                print(ds)
                answer = input(
                    "You are about the remove this dataset ({Id}). Do you want to continue? (y/N)".format(Id=Id))
                if answer.lower() in ["y", "yes"]:
                    store.delete(Id)
                else:
                    print("Skipping, will not remove {Id}".format(Id=Id))

    def features(self):
        """List features for a dataset command"""
        parser = core_parser(
            prog="kosh features",
            description='List features for (some) dataset(s)')
        parser.add_argument("--ids", "-i", help="ids of datasets to list features from",
                            nargs="*", required=True, action="append")
        args = parser.parse_args(sys.argv[2:])
        datasets = []
        for i in args.ids:
            datasets += i

        store = kosh.KoshStore(
            db_uri=args.store, dataset_record_type=args.dataset_record_type)
        for Id in datasets:
            ds = store.open(Id)
            print("\nDataset: {}:\n\t {}".format(Id, ds.list_features()))

    def extract(self):
        """Extract feature from dataset"""
        parser = core_parser(
            prog="kosh extract",
            description='Extract features from a dataset')
        parser.add_argument(
            "--id", "-i", help="id of datasets to extract features from", required=True)
        parser.add_argument(
            "--features", "-f", help="features to extract", nargs="*", action="append")
        parser.add_argument(
            "--format", "-F", help="format to extract to", default="numpy")
        parser.add_argument("--dump", help="Dump to file")
        args, extract_terms = parser.parse_known_args(sys.argv[2:])
        extract_terms = parse_metadata(extract_terms)
        store = kosh.KoshStore(
            db_uri=args.store, dataset_record_type=args.dataset_record_type)
        ds = store.open(args.id)
        features = []
        if args.features is not None:
            for f in args.features:
                features += f
        else:
            features = ds.list_features()

        if args.dump is not None:
            out = open(args.dump, "wb")
        for feat in features:
            data = ds.get(feat, format=args.format, **extract_terms)
            if args.dump is not None:
                try:
                    import numpy
                    numpy.save(out, data)
                except Exception:
                    print(
                        "Could not save feature {feat} to file: {args.dump}".format(feat=feat, args=args))
            else:
                print(data)

    def print(self):
        """print dataset(s)"""
        parser = core_parser(
            prog="kosh print",
            description='Print information about a dataset')
        parser.add_argument("--ids", "-i", help="ids of datasets to print",
                            nargs="*", required=True, action="append")
        args = parser.parse_args(sys.argv[2:])
        datasets = []
        for i in args.ids:
            datasets += i

        store = kosh.KoshStore(
            db_uri=args.store, dataset_record_type=args.dataset_record_type)
        for Id in datasets:
            ds = store.open(Id)
            print(ds)
            print(
                "=======================================================================")

    def associate(self):
        """Associate uri with dataset command"""
        parser = core_parser(
            prog="kosh associate",
            description="Associate a (set of) files with a dataset")
        parser.add_argument(
            "--id", "-i", help="id of datasets to which file(s) will be associated", required=True)
        parser.add_argument("--uri", "-u", help="uri(s) to associate with dataset",
                            nargs="*", required=True, action="append")
        parser.add_argument(
            "--mime_type", "-m", help="mime type of the uri(s) same for all", required=True)
        args = parser.parse_args(sys.argv[2:])
        uris = []
        for u in args.uri:
            uris += u
        store = kosh.KoshStore(
            db_uri=args.store, dataset_record_type=args.dataset_record_type)
        ds = store.open(args.id)
        for u in uris:
            ds.associate(u, mime_type=args.mime_type)

    def dissociate(self):
        """dissociate uri from dataset command"""
        parser = core_parser(
            prog="kosh dissociate",
            description="dissociate a (set of) file(s) from a dataset")
        parser.add_argument(
            "--id", "-i", help="id of datasets from which file(s) will be dissociated", required=True)
        parser.add_argument("--uri", "-u", help="uri(s) to dissociate from dataset",
                            nargs="*", required=True, action="append")
        args = parser.parse_args(sys.argv[2:])
        uris = []
        for u in args.uri:
            uris += u
        store = kosh.KoshStore(
            db_uri=args.store, dataset_record_type=args.dataset_record_type)
        ds = store.open(args.id)
        for u in uris:
            ds.dissociate(u)

    def mv(self):
        """mv files command"""
        self._mv_cp_("mv")

    def cp(self):
        """cp files command"""
        self._mv_cp_("cp")

    def tar(self):
        """tar files command"""
        parser = argparse.ArgumentParser(
            prog="kosh tar",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="(un)tar files and the dataset they're associated with in selected Kosh store(s)",
            epilog="Kosh version {kosh.__version__}".format(kosh=kosh))
        parser.add_argument("--stores", "--store", "-s", required=True,
                            help="Kosh store(s) to use", action="append")
        parser.add_argument("--dataset_record_type", default="dataset",
                            help="type used by sina db that Kosh will recognize as dataset")
        parser.add_argument("-f", "--file", help="tar file", required=True)
        parser.add_argument(
            "--no_absolute_path",
            action="store_true",
            help="Do not use absolute path when searching stores")
        parser.add_argument("--dataset_matching_attributes", default=["name", ],
                            help="List of attributes used to identify if two datasets are identical",
                            type=ast.literal_eval)
        args, opts = parser.parse_known_args(sys.argv[2:])

        # Ok are we creating or extracting?
        extract = False
        create = False
        if "x" in opts[0]:
            extract = True
            if "v" not in opts[0]:
                opts[0] += "v"
        if "c" in opts[0]:
            create = True

        if create == extract:
            raise RuntimeError(
                "Could not determine if you want to create or extract the archive, aborting!")

        # Open the stores
        stores = open_stores(args.stores, args.dataset_record_type)

        if create:
            # Because we cannot add the exported dataset file to a compressed archive
            # We need to figure out the list of files first
            # They should all be in opts and the tar file is not because of -f
            # option
            tarred_files = get_all_files(opts)

            # Prpare dictionar to hold list of datasets to epxort (per store)
            store_datasets = {}
            for store in stores:
                store_datasets[store.db_uri] = []

            # for each tar file let's see if it's associated to a/many
            # datset(s) in the store
            for filename in tarred_files:
                if not args.no_absolute_path:
                    filename = os.path.abspath(filename)
                for store in stores:
                    store_datasets[store.db_uri] += store.search(
                        file=filename, ids_only=True)

            # Ok now we need to export all the datasets to a file
            # First item is the root from where we ran the command (untar will
            # need this)
            datasets_jsons = [os.getcwd(), ]
            for store in stores:
                for Id in set(
                        store_datasets[store.db_uri]):  # loop throuh datasets but make sure only once
                    datasets_jsons.append(store.export_dataset(Id))

            # Ok let's dump this
            tmp_json = tempfile.NamedTemporaryFile(prefix="__kosh_export__",
                                                   suffix=".json",
                                                   dir=os.getcwd(),
                                                   mode="w")
            json.dump(datasets_jsons, tmp_json)
            # Make sure it's all in the file before tarring it
            tmp_json.file.flush()

            # Let's tar this!
            cmd = "tar {} {} -f {}".format(" ".join(opts),
                                           os.path.basename(tmp_json.name), args.file)
        else:  # ok we are extracting
            cmd = "tar {} -f {}".format(" ".join(opts), args.file)

        p, out, err = process_cmd(cmd)

        if p.returncode != 0:
            raise RuntimeError(
                "Could not run tar cmd: {}\nReceived error: {}".format(
                    cmd, err.decode()))

        if extract:
            # ok we extracted that's nice
            # Now let's populate the stores

            # Step 1 figure out the json file that contains our datsets
            filenames = out.decode().split("\n")
            # tar removes leading slah from full path
            slashed_filenames = ["/" + x for x in filenames]
            for filename in filenames:
                if filename[:15] == "__kosh_export__" and filename[-5:] == ".json":
                    break
            with open(filename) as f:
                datasets = json.load(f)

            # Step 2 recover the root path from where the tar was made
            # And our guessed tarrred files
            root = datasets.pop(0)
            root_filenames = [os.path.join(root, x) for x in filenames]

            # Step 3 let's put these datasets into the stores
            for dataset in datasets:
                # Let's try to recover the correct path now..
                delete_them = []
                for index, associated in enumerate(dataset["associated"]):
                    uri = associated["uri"]
                    if uri in filenames:
                        new_uri = os.path.join(os.getcwd(), uri)
                    elif uri in slashed_filenames:  # tar removes leading /
                        new_uri = os.path.join(
                            os.getcwd(), filenames[slashed_filenames.index(uri)])
                    elif uri in root_filenames:
                        new_uri = os.path.join(
                            os.getcwd(), filenames[root_filenames.index(uri)])
                    else:
                        if not os.path.exists(uri):
                            delete_them.append(index)
                        new_uri = None
                    associated["uri"] = new_uri

                # Yank uris that do not exists in this filesystem
                for index in delete_them[::-1]:
                    dataset["associated"].pop(index)

                # Add dataset to store(s)
                for store in stores:
                    store.import_dataset(
                        dataset, args.dataset_matching_attributes)

    def rm(self):
        """rm files command"""
        parser = argparse.ArgumentParser(
            prog="kosh rm",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="delete files and update their uris in selected Kosh store(s)",
            epilog="Kosh version {kosh.__version__}".format(kosh=kosh))
        parser.add_argument("--stores", "--store", "-s", required=True,
                            help="Kosh store(s) to use", action="append")
        parser.add_argument("--sources", "--source", "-S", "-f", "--file", "--files",
                            help="source (files or directories) to move",
                            action="append", nargs="+", required=True)
        parser.add_argument("--dataset_record_type", default="dataset",
                            help="type used by sina db that Kosh will recognize as dataset")
        args, opts = parser.parse_known_args(sys.argv[2:])

        # Ok first step is to list files to yank
        sources = []
        for source in args.sources:
            if isinstance(source, list):
                sources += source
            else:
                sources.append(source)
        files = []
        for source in sources:
            files += glob.glob(source)

        # Ok now we need abspaths
        files = [os.path.abspath(f) for f in files]

        # Finally we need to see if there are directories in this
        del_indices = []
        for i, f in enumerate(list(files)):
            if os.path.isdir(f):
                del_indices.append(i)
                for root, dirs, filenames in os.walk(f):
                    for filename in filenames:
                        files.append(os.path.join(root, filename))
        # remove dirs
        for index in del_indices[::-1]:
            files.pop(index)

        # Ok now open the stores
        stores = open_stores(args.stores, args.dataset_record_type)

        # And finally delete the files
        for filename in files:
            try:
                os.remove(filename)
            except Exception:
                continue

            # it worked let's remove it from stores
            for store in stores:
                datasets = store.search(file=filename)
                for dataset in datasets:
                    dataset.dissociate(filename)

    def fast_sha(self):
        """print fast_sha Kosh would compute for a list of files"""
        parser = argparse.ArgumentParser(
            prog="kosh fast_sha",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="print short shas that kosh will compute for some sources",
            epilog="Kosh version {kosh.__version__}".format(kosh=kosh))
        args, opts = parser.parse_known_args(sys.argv[2:])

        files = get_all_files(opts)
        for filename in files:
            sha = kosh.utils.compute_fast_sha(filename)
            print("{} {}".format(sha, filename))

    def reassociate(self):
        """reassociate files with datasets"""
        parser = argparse.ArgumentParser(
            prog="kosh reassociate",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="reassociate uris in selected Kosh store(s)",
            epilog="Kosh version {kosh.__version__}".format(kosh=kosh))
        parser.add_argument("--stores", "--store", "-s", required=True,
                            help="Kosh store(s) to use", action="append")
        parser.add_argument("--new_uris", "--new_uri", "-n",
                            help="new uri(s) to use instead of old ones",
                            action="append", nargs="+", required=True)
        parser.add_argument("--original_uris", "--original_uri",
                            help="uri or sha in store that we wish to replace," +
                            " if not passed then will use short sha(s) of new_uri(s)",
                            action="append", nargs="+", default=None)
        parser.add_argument(
            "--no_absolute_path",
            action="store_true",
            help="Do not change paths to absolute path")
        parser.add_argument("--dataset_record_type", "-d", default="dataset",
                            help="type used by sina db that Kosh will recognize as dataset")

        args, opts = parser.parse_known_args(sys.argv[2:])

        new_uris = find_files_from_list(args.new_uris)

        if args.original_uris is not None:
            original_uris = find_files_from_list(args.original_uris)
            if len(original_uris) != len(new_uris):
                raise RuntimeError(
                    "you are trying to reassociate {} uris to {} new uris. Aborting!".format(
                        len(original_uris), len(new_uris)))
        else:
            original_uris = None

        for store_uri in args.stores:
            store = kosh.KoshStore(
                store_uri, dataset_record_type=args.dataset_record_type)
            for index, target in enumerate(new_uris):
                if original_uris is None:
                    store.reassociate(
                        target, absolute_path=not args.no_absolute_path)
                else:
                    store.reassociate(
                        target,
                        original_uris[index],
                        absolute_path=not args.no_absolute_path)

    def _mv_cp_(self, command):
        """core function to implement mv and cp"""
        if command == "mv":
            command_str = "move"
        elif command == "cp":
            command_str = "copy"
        else:
            raise ValueError("Unknown command {}".format(command))

        parser = argparse.ArgumentParser(
            prog="kosh {}".format(command),
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="{} files and update their uris in selected Kosh store(s)".format(
                command_str),
            epilog="Kosh version {kosh.__version__}".format(kosh=kosh))
        parser.add_argument("--stores", "--store", "-s", required=True,
                            help="Kosh store(s) to use", action="append")
        if command != "rm":
            parser.add_argument("--destination-stores", "--destination-store",
                                "--destination_stores", "--destination_store",
                                help="Kosh store(s) to use", action="append", default=[])
        parser.add_argument("--sources", "--source", "-S", "-f", "--file", "--files",
                            help="source (files or directories) to move",
                            action="append", nargs="+", required=True)
        parser.add_argument("--dataset_record_type", "-d", default="dataset",
                            help="type used by sina db that Kosh will recognize as dataset")
        parser.add_argument("--dataset_matching_attributes", default=["name", ],
                            help="List of attributes used to identify if two datasets are identical",
                            type=ast.literal_eval)
        if command != "rm":
            parser.add_argument("--destination",
                                help="destination (file or directory) name", required=True)
        parser.add_argument("--version", "-v", action="store_true",
                            help="print version and exit")
        args, opts = parser.parse_known_args(sys.argv[2:])
        files = []
        for source in args.sources:
            if isinstance(source, list):
                files += source
            else:
                files.append(source)
        for filename in files:
            if "://" in filename:
                if len(files) > 1:
                    raise ValueError(
                        "URIS can only be moved one at a time `kosh mv --stores=[...] ur1 ur2")

        if command in ["cp", "mv"]:
            target = args.destination
        else:
            target = "/tmp"

        if command == "mv":   # Move only works on local files
            if is_remote(target) or sum([is_remote(x) for x in files]) > 0:
                raise ValueError("kosh mv only works on local files")

        sources, targets = find_sources_and_targets(opts, files, target)

        if command == "rm":
            targets = ["", ] * len(sources)

        # for i, source in enumerate(sources):
        #     print("Doing: {} to {}".format(source,targets[i]))

        origin_stores = open_stores(args.stores, args.dataset_record_type)

        dest_stores = open_stores(
            args.destination_stores,
            args.dataset_record_type)
        # Ok now let's prepare the actual command
        if command in ["mv", "cp"]:
            cmd = "rsync -v -r " + " ".join(opts)
        elif command in ["rm"]:
            cmd = "rm " + " ".join(opts)
        if command == "mv":
            # let's yank the source
            cmd += " --remove-source-files "
        rsync_ran = []
        for i, source in enumerate(sources):
            for o_store in origin_stores:
                datasets = o_store.search(file=source)
                for dataset in datasets:
                    if command == "mv":
                        associated_uris = dataset.search(uri=source)
                        for associated in associated_uris:
                            associated.uri = targets[i]
                    else:
                        exported = dataset.export()
                        # Ok we need to update the uri to point t the new
                        # target
                        delte_these = []
                        for indx, a in enumerate(exported["associated"]):
                            if a["uri"] == source:
                                a["uri"] = targets[i]
                            else:
                                delte_these.append(indx)
                        for indx in delte_these[::-1]:
                            del(exported["associated"][indx])

                        for d_store in dest_stores:
                            d_store.import_dataset(
                                exported, args.dataset_matching_attributes)

            # Now let's run the command and see if it worked
            # But only if not ran for directory before
            if os.path.dirname(source) + "/" not in sources[:i]:
                skip_it = False
                for ran in rsync_ran:
                    if ran in source:
                        skip_it = True
                        break
                if skip_it:
                    continue
                rsync_ran.append(source)
                cmd_run = cmd + " {} {}".format(source, targets[i])
                p, o, e = process_cmd(cmd_run)
                if p.returncode != 0:  # Failed!
                    raise RuntimeError(
                        "Error runnning command {}, aborting!\n{}".format(
                            cmd_run, e.decode()))

            # Ok it's good let's sync the stores
            for store in origin_stores + dest_stores:
                store.sync()

        # closes stores and send them back to remote if necessary
        close_stores(origin_stores, args.stores)
        close_stores(dest_stores, args.destination_stores)


def is_remote(path):
    """Determine if a uri is located on a remote server
    First figures out if there is a ':' in the path (e.g user@host:/path)
    Then looks if there is a single @ in part preceding the first ':'
    :param path: uri/path
    :type path: str
    :return: True if path points to remote server, False otherwise
    :rtype: bool
    """
    dot = path.split(":")[0]
    at = dot.split("@")
    if len(at) == 1:
        return False
    return True


def get_realpath_and_status(source):
    """figure out if a path is:
        local or remote
        a dir or not
        if it exists or not
    works with remote paths as well.
    Return absolute path on host to directory where file exist or directory
    :param source: path to use
    :type source: str
    :return: abs path on host, is_it_remote, is_it_a_directory, does_it_exist
    :rtype: str, bool, bool, bool
    """
    is_dir_cmd = "if [ -d {} ]; then echo -e 1 ; else echo -e 0 ;  fi ;"
    realpath_cmd = "realpath {}"
    sp = source.split("@")
    if len(sp) > 1:
        is_remote = True
        user = sp[0]
        machine = sp[1].split(":")[0]
        files = ":".join("@".join(sp[1:]).split(":")[1:])
        # Find user and machine
        ssh_prefix = "ssh {}@{} ".format(user, machine)
        is_dir_cmd = ssh_prefix + "'{}'".format(is_dir_cmd)
        realpath_cmd = ssh_prefix + "'{}'".format(realpath_cmd)
    else:
        is_remote = False
        files = source
    is_dir_cmd = is_dir_cmd.format(files)
    realpath_cmd = realpath_cmd.format(files)
    if "*" in is_dir_cmd or "?" in is_dir_cmd:
        is_dir = 0
        is_file = 1
    else:
        _, o, _ = process_cmd(is_dir_cmd, use_shell=True)
        o = o.decode().split("\n")
        is_dir = int(o[0])
        cmd = is_dir_cmd.replace("-d", "-f")
        _, o, _ = process_cmd(cmd, use_shell=True)
        o = o.decode().split("\n")
        is_file = int(o[0])
    if is_dir == 0 and is_file == 0:
        # Ok it does not exist
        exists = False
    else:
        exists = True
    _, realpath, _ = process_cmd(realpath_cmd.format(files), use_shell=True)
    return realpath.decode().strip().split("\n")[0], is_remote, is_dir, exists


def find_depth(path):
    """Given a path returns how level of directories this is in
    :param path: path to scan
    :type path: str
    :return: number of directories in which the file is in this path
    :rtype: int
    """
    depth = 0
    tmp = os.path.split(path)
    while tmp[-1] != "":
        depth += 1
        tmp = os.path.split(tmp[0])
    return depth - 1


def find_sources_and_targets(options, sources, target):
    """Given a list of sources (files, dir, patterns) and a target destination,
    runs 'rsync' between these to obtain the list of files being touched
    :param options: option to send to rsync
    :type options: list
    :param sources: list of sources files, dirs or patterns
    :type sources: list
    :param target: target file or directory
    :type target: str
    :return: List of sources and there matching path after cp/mv
    :rtype: list, list
    """
    target_realpath, is_target_remote, is_target_dir, target_exists = get_realpath_and_status(
        target)
    source_uris = []
    target_uris = []
    for source in sources:
        source_realpath, is_source_remote, is_source_dir, source_exists = get_realpath_and_status(
            source)
        if len(sources) > 1 and not is_source_dir and not target_exists:
            raise ValueError(
                "Destination ({}) does not exists and you're trying to send multiple sources to it".format(target))
        if not source_exists:
            raise RuntimeError("Source {} does not exists".format(source))
        # if len(sources)==1 and not is_target_dir and not is_target_file and is_source_dir:
        #    raise ValueError("Destination does not exists and you're trying to send multiple sources to it")
        cmd = "rsync -v --dry-run -r" + " ".join(options)
        cmd += " " + source + " " + target
        p, o, e = process_cmd(cmd, use_shell=True)
        rsync_dryrun_out_lines = o.decode().split("\n")

        found_a_dir_to_rsync = False
        for i, ln in enumerate(rsync_dryrun_out_lines):
            if i == 0:
                depth = find_depth(ln) + 1
                counter = 0
                non_base = []
                abs_path = source_realpath
                while counter != depth:
                    non_base.insert(0, os.path.basename(abs_path))
                    abs_path = os.path.dirname(abs_path)
                    counter += 1
                non_base = os.path.join(*non_base)
            if len(ln.strip()) == 0:
                break
            elif len(shlex.split(ln.strip())) == 1:
                if ln.strip()[-1] == "/":
                    if found_a_dir_to_rsync is False:
                        source_uris.append(os.path.join(
                            abs_path, ln.strip()[:-1]))
                        found_a_dir_to_rsync = True
                    else:
                        continue
                else:
                    source_uris.append(os.path.join(abs_path, ln.strip()))
                if target_exists:
                    if ln.strip()[-1] == "/":
                        # it's a dir
                        target_uris.append(target_realpath)
                    else:
                        target_uris.append(os.path.join(
                            target_realpath, ln.strip()))
                else:
                    if ln.strip()[-1] == "/" or not is_source_dir:
                        target_uris.append(target_realpath)
                    else:
                        target_uris.append(os.path.join(
                            target_realpath, ln.strip()))
    return source_uris, target_uris


if __name__ == '__main__':
    KoshCmd()
