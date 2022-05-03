#!/bin/sh sbang
#!/usr/bin/env python
# This implements Kosh's CLI
from __future__ import print_function
import argparse
from kosh.utils import merge_datasets_handler
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
import six
import sys


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
    parser.add_argument("--version", action="store_true",
                        help="print version and exit")
    return parser


def parse_metadata(terms):
    """
    Parse metadata for Kosh  queries
    param=value / param>value, etc...
    :param terms: list of strings conatining name/operator/value
    :term terms: list of str
    :return: Dictionary with name as key and matching sina find object as value
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
        if not sys.platform.startswith("win"):
            proc = Popen(shell, stdin=PIPE, stdout=PIPE, stderr=PIPE)
            o, e = proc.communicate(command.encode())
        else:
            proc = Popen(command, stdout=PIPE, stderr=PIPE, shell=True)
            o, e = proc.communicate()
            print("COMMNAD:" , command)
            print("OUT:", o.decode())
            print("ERR:", e.decode())
    else:
        if not sys.platform.startswith("win"):
            command = shlex.split(command)
        proc = Popen(command, stdout=PIPE, stderr=PIPE)
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
    if isinstance(dataset_record_type, six.string_types):
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

        # search is deprecated let's not list it
        index = commands.find("search")
        if index > -1:
            commands = commands[:index] + commands[index+8:]  # 8 because of \n\t

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
        """
        Deprecated use find
        """
        warnings.warn(DeprecationWarning, "The 'search' command is deprecated and now called `find`.\n"\
                      "Please update your code to use `find` as `search` might disappear in the future")
        return self.find()

    def find(self):
        """find in a store command"""
        parser = core_parser(
            description='Find datasets in store that are matching metadata in form key=value')
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
        ids = store.find(**metadata)
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
        some_att=some_val will only dissociate non-existing files associated and having the attribute "some_att" with value of "some_val"""
        parser = core_parser(
            prog="kosh clean",
            description="""Cleanup a store from references to dead files
        You can filter associated object by matching metadata in form key=value
        e.g mime_type=hdf5 will only dissociate non-existing files associated with mime_type hdf5
        some_att=some_val will only dissociate non-existing files associated and having the attribute
        'some_att' with value of 'some_val'""")
        parser.add_argument("--dry-run", "--rehearsal", "-r", "-D", help="Dry run only, list ids of dataset that would be cleaned up and path of files", action='store_true')
        parser.add_argument("--interactive", "-i", help="interactive mode, ask before dissociating", action="store_true")
        args, search_terms = parser.parse_known_args(sys.argv[2:])
        metadata = parse_metadata(search_terms)
        store = kosh.KoshStore(db_uri=args.store,
                               dataset_record_type=args.dataset_record_type)
        ids = store.find(ids_only=True)
        for Id in ids:
            ds = store.open(Id)
            missings = ds.cleanup_files(dry_run=args.dry_run, interactive=args.interactive, **metadata)
            if len(missings) != 0:
                if not args.interactive:  # already printed in interactive
                    print(ds)
                for uri in missings:
                    associated = list(ds.find(uri=uri))
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
        ds = store.create(id=args.id, metadata=metadata)
        print(ds.id)

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
        if sys.platform.startswith("win"):
            raise SystemError("you cannot use kosh mv on a windows system, please try to tar/untar or manually move the files and use reassociate")
        self._mv_cp_("mv")

    def cp(self):
        if sys.platform.startswith("win"):
            raise SystemError("you cannot use kosh cp on a windows system, please manually cp the files and associate them")
        """cp files command"""
        self._mv_cp_("cp")

    def tar(self):
        """tar files"""
        self._tar("tar", "Uses `tar` to (un)tar files and the dataset they're associated with in selected Kosh store(s)")

    def htar(self):
        """tar files using htar"""
        self._tar("htar", "Uses htar to (un)tar files and the dataset they're associated with in selected Kosh store(s)")

    def _tar(self, tar_command, description):
        """tar files command"""
        parser = argparse.ArgumentParser(
            prog="kosh tar",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description=description,
            epilog="Kosh version {kosh.__version__}".format(kosh=kosh))
        parser.add_argument("--stores", "--store", "-s", required=True,
                            help="Kosh store(s) to use", action="append")
        parser.add_argument("--dataset_record_type", default="dataset",
                            help="record type used by Kosh when adding datasets to Sina database")
        parser.add_argument("-f", "--file", help="tar file", required=True)
        parser.add_argument(
            "--no_absolute_path",
            action="store_true",
            help="Do not use absolute path when searching stores")
        parser.add_argument("--dataset_matching_attributes", default=["name", ],
                            help="List of attributes used to identify if two datasets are identical",
                            type=ast.literal_eval)
        parser.add_argument("--merge_strategy", help="When importing dataset, how do we handle conflict", default=None, choices=["conservative", "preserve", "overwrite"])
        args, opts = parser.parse_known_args(sys.argv[2:])

        # Ok are we creating or extracting?
        extract = False
        create = False
        if "x" in opts[0] or "-x" in opts:
            extract = True
            if "v" not in opts[0] and "-v" not in opts:
                opts.append("-v")
        if "c" in opts[0] or "-c" in opts:
            create = True

        if "t" in opts[0] or "-t" in opts:
            raise ValueError("t (test archive) option is not supported yet")

        if create == extract:
            raise RuntimeError(
                "Could not determine if you want to create or extract the archive, aborting!")

        # Open the stores
        stores = open_stores(args.stores, args.dataset_record_type)

        if create:
            clean_json = True  # windows needs us to del manually
            # Because we cannot add the exported dataset file to a compressed archive
            # We need to figure out the list of files first
            # They should all be in opts and the tar file is not because of -f
            # option
            no_tarred_files = False
            tarred_files = get_all_files(opts)
            if tarred_files == []:
                no_tarred_files = True
                # Ok user did not pass files
                # that means we need to tar
                # all files in store
                for store in stores:
                    recs = store.get_sina_records()
                    file_type = recs.get("__kosh_store_info__")["data"]["sources_type"]["value"]
                    # Could check the file exists
                    tarred_files += [x["data"]["uri"]["value"] for x in recs.find_with_type(file_type)]

            # Prepare dictionary to hold list of datasets to export (per store)
            store_datasets = {}
            for store in stores:
                store_datasets[store.db_uri] = []

            # for each tar file let's see if it's associated to a/many
            # dataset(s) in the store
            for filename in tarred_files:
                if not args.no_absolute_path:
                    filename = os.path.abspath(filename)
                for store in stores:
                    recs = store.get_sina_records()
                    file_type = recs.get("__kosh_store_info__")["data"]["sources_type"]["value"]
                    store_ds = [x["data"]["associated"]["value"] for x in recs.find(data={"uri":filename}, types=[file_type,])]
                    for ds in store_ds:
                        store_datasets[store.db_uri] += ds

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
                                                   dir=os.path.abspath(os.path.dirname(args.file)),
                                                   mode="w",
                                                   delete=False)
            json.dump(datasets_jsons, tmp_json)
            # Make sure it's all in the file before tarring it
            tmp_json.file.flush()
            if no_tarred_files:
                tarred_files = [os.path.relpath(x) for x in tarred_files]
                opts += tarred_files

            tmp_json.file.close()
            # Let's tar this!
            cmd = "{} -f {} {} {}".format(tar_command, args.file, " ".join(opts), tmp_json.name)
        else:  # ok we are extracting
            clean_json = False  # no json created
            cmd = "{} -f {} {}".format(tar_command, args.file, " ".join(opts))

        p, out, err = process_cmd(cmd)
        if sys.platform.startswith("win"):  # windows tar put message in err
            err, out = out, err
        if clean_json:
            os.remove(tmp_json.name)
        if p.returncode != 0:
            raise RuntimeError(
                "Could not run {} cmd: {}\nReceived error: {}".format(
                    tar_command, cmd, err.decode()))

        if extract:
            # ok we extracted that's nice
            # Now let's populate the stores

            # Step 1 figure out the json file that contains our datasets
            filenames = out.decode().split("\n")
            if sys.platform.startswith("win"):
                filenames = [filename.split()[1] for filename in filenames if len(filename.split()) > 1]
            if "HTAR" in filenames[0]:
                # htar used
                filenames = filenames[:-3]  # last 3 lines are nothing
                filenames = [x.split(",")[0].split()[-1].strip() for x in filenames]
            # tar removes leading slah from full path
            slashed_filenames = ["/" + x for x in filenames]
            for ifile, filename in enumerate(filenames):
                base = os.path.basename(filename)
                if base.startswith("__kosh_export__") and base.endswith(".json"):
                    break
            with open(filename) as f:
                datasets = json.load(f)

            os.remove(filename)

            # Step 2 recover the root path from where the tar was made
            # And our guessed tarrred files
            root = datasets.pop(0)
            root_filenames = [os.path.join(root, x) for x in filenames]

            # Step 3 let's put these datasets into the stores
            orphans = []
            for dataset in datasets:
                # Let's try to recover the correct path now..
                delete_them = []
                for index, associated in enumerate(dataset["records"][1:]):
                    uri = associated["data"]["uri"]["value"]
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
                            delete_them.append(index + 1)
                        else:
                            orphans.append(uri)
                        new_uri = uri
                    associated["data"]["uri"]["value"] = new_uri


                # Yank uris that do not exists in this filesystem
                for index in delete_them[::-1]:
                    dataset["records"].pop(index)

                # Add dataset to store(s)
                for store in stores:
                    store.import_dataset(
                        dataset, args.dataset_matching_attributes,
                        merge_handler=args.merge_strategy)

            # Trying to reassociate these orphan files
            # path aliases might cause them to appear
            matches = {}
            for orphan in orphans:
                matches[orphan] = []
                # now let's try to find a possible match
                for myfile in files:
                    if len(myfile) < 2:
                        continue
                    if orphan.endswith(myfile):
                        matches[orphan].append(myfile)

            # go through the matches
            for match in matches:
                for dataset in s.find(file=match):
                    # If fast_sha changed we're hosed
                    # Trying to fix this
                    dataset.cleanup_files(clean_fastsha=True)
                    for possible in matches[match]:
                        dataset.reassociate(possible)

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
                datasets = store.find(file=filename)
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
        parser.add_argument("--version", action="store_true",
                            help="print version and exit")
        parser.add_argument("--merge_strategy", help="When importing dataset, how do we handle conflict", default=None, choices=["conservative", "preserve", "overwrite"])
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
                datasets = o_store.find(file=source)
                for dataset in datasets:
                    if command == "mv":
                        associated_uris = dataset.find(uri=source)
                        for associated in associated_uris:
                            associated.uri = targets[i]
                        # we also need to update the file section of our record
                        rec = dataset.get_record()
                        rec["files"][targets[i]] = rec["files"][source]
                        del(rec["files"][source])
                        o_store.get_sina_records().update(rec)
                    else:
                        exported = dataset.export()
                        # Ok we need to update the uri to point to the new
                        # target
                        delte_these = []
                        for indx, a in enumerate(exported["records"][1:]):
                            if a["data"]["uri"]["value"] == source:
                                a["data"]["uri"]["value"] = targets[i]
                            else:
                                delte_these.append(indx + 1)
                        for indx in delte_these[::-1]:
                            del(exported["records"][indx])

                        for d_store in dest_stores:
                            d_store.import_dataset(
                                exported, args.dataset_matching_attributes,
                                merge_handler=args.merge_strategy)

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

    def create_new_db(self):
        """Creates a Kosh store"""
        parser = argparse.ArgumentParser(
            prog="kosh create_new_db",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Creates a new Kosh store",
            epilog="Kosh version {kosh.__version__}".format(kosh=kosh))
        parser.add_argument("--uri", "-u", help="path to database", required=True) 
        parser.add_argument("--database", "--db", "-d", help="Database type to use as backend", choices=["sql", "cass"], default="sql")
        parser.add_argument("--token", "-t", help="Token to use (for Cassandra databases)", default="")
        parser.add_argument("--keyspace", "-k", help="keyspace to use (for Cassandra databases)")
        parser.add_argument("--cluster", "-c", help="cluster to use (for Cassandra databases)")

        args = parser.parse_args(sys.argv[2:])

        kosh.create_new_db(args.uri, db=args.database,
                           token=args.token, keyspace=args.keyspace, cluster=args.cluster)
    
    def create(self):
        """Creates a Kosh dataset in a store"""
        parser = core_parser(
            description='Create a dataset in the store with matching metadata in form key=value')
        args, user_params = parser.parse_known_args(sys.argv[2:])

        params = {}
        index = 0
        while index < len(user_params):
            term = user_params[index]
            sp = term.split("=")
            if len(sp) > 1:
                params[sp[0]] = eval(sp[1])
                index += 1
            else:
                params[sp[0]] = eval(user_params[index+1])
                index += 2

        print("Adding ds to: {}".format(args.store))
        store = kosh.KoshStore(args.store)
        store.create(metadata=params)



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
