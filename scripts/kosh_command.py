#!/usr/bin/env python

import argparse
import kosh
import sys
from sina.utils import DataRange

def core_parser(description,
                usage=None):
    """
    Return the core parser with arguments common to all operations
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=description,
        usage=usage)
    parser.add_argument("--store", "-s", required=True,
                        help="Kosh store to use")
    parser.add_argument("--dataset_record_type", "-d", default="dataset",
                        help="type used by sina db that Kosh will recognize as dataset")
    return parser

def parse_metadata(terms):
    metadata = {}
    for term in terms:
        found = False
        for operator in ["=", "<", ">","<=", ">="]:
            sp = term.split(operator)
            if len(sp) != 2:
                continue
            found = True
            key = sp[0]
            try:
                value = eval(sp[1])  # converts to int/float/DataRange if possible
            except Exception:
                value = sp[1]
            if operator == "<=":
                value = DataRange(min=value)
            elif operator == "<":
                value = DataRange(min=value, min_inclusive=False)
            elif operator == ">=":
                value = DataRange(max=value, max_inclusive=True)
            elif operator == ">":
                value = DataRange(max=value, max_inclusive=False)
            metadata[key]=value

        if not found:
            raise ValueError("Metadata must be in form 'key=value'")
    return metadata


class KoshCmd(object):
    def __init__(self):
        commands = "".join(["" if k[0]=="_" else "\n\t"+k for k in sorted(dir(self))])
        parser = core_parser(
            description='Execute kosh operations',
            usage=f'''kosh <command> [<args>]

Available commands are:
    {commands}
''')
        parser.add_argument('command', help='Subcommand to run')
        # parse_args defaults to [1:] for args, but you need to
        # exclude the rest of the args too, or validation will fail
        args = parser.parse_args(sys.argv[1:2] + ["-s","blah"])
        if not hasattr(self, args.command) or args.command[0] == "_":
            print(f'Unrecognized command: {args.command}')
            print(f'Known commands: {" ".join(["" if k[0]=="_" else k for k in dir(self)])}')
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def search(self):
        parser = core_parser(
            description='Search Kosh store for datasets matching metadata in form key=value')
        parser.add_argument("--print", "-p", help="print each dataset info", action="store_true")
        args, search_terms = parser.parse_known_args(sys.argv[2:])
        metadata = parse_metadata(search_terms)
        store = kosh.KoshStore(db_uri=args.store)
        ids = store.search(**metadata, ids_only=True)
        if args.print:
            for Id in ids:
                ds = store.open(Id)
                print(ds)
                print("=======================================================================")
        else:
            print("\n".join(ids))
 
    def print(self):
        parser = core_parser(
            description='Print information about a dataset')
        parser.add_argument("--ids", "-i", help="ids of datsets to print", nargs="*", required=True, action="append")
        args = parser.parse_args(sys.argv[2:])
        datasets = []
        for i in args.ids:
            datasets += i

        store = kosh.KoshStore(db_uri=args.store)
        for Id in datasets:
                ds = store.open(Id)
                print(ds)
                print("=======================================================================")

    def associate(self):
        parser = core_parser(description="Associate a file with a dataset")
        parser.add_argument("--ids", "-i", help="ids of datsets to print", nargs="*", required=True, action="append")
        args = parser.parse_args(sys.argv[2:])
        datasets = []
        for i in args.ids:
            datasets += i
        print("Associate")


if __name__ == '__main__':
    KoshCmd()
