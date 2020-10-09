from subprocess import Popen, PIPE
import os
import shlex
import sys
import kosh
import hashlib
import numpy
import copy
from collections import OrderedDict


def compute_fast_sha(uri, n_samples=10):
    """Compute a fast 'almost' unique identifier for a given uri
    Assumes the uri is a path to a file, otherwise simply return hexdigest of md5 on the uri string

    If uri path is valid the 'fast' sha is used by creating an hashlib from
    * file size
    * file first 2kb
    * file last 2kb
    * 2k samples read from `n_samples` evenly spaced in the file

    Warning if size is unchanged and data is changed somewhere else than those samples the sha will be identical
    :param uri: URI to compute fast_sha on
    :type uri: str
    :param n_samples: Number of samples to extract from uri (in addition to beg and end of file)
    :type n_sampe: int
    :return sha: hexdigested sha
    :rtype: str
    """
    if not os.path.exists(uri):
        sha = hashlib.sha256(uri.encode())
        return sha.hexdigest()
    with open(uri, "rb") as f:
        stats = os.fstat(f.fileno())
        size = stats.st_size
        sha = hashlib.sha256("{}".format(size).encode())
        # Create list of start read
        positions = [int(max(x, 0))
                     for x in numpy.linspace(0, size - 2048, n_samples + 2)]
        prev = -1
        for pos in positions:
            # Small file will have multiple times the same bit to read
            if pos != prev:
                # Go there
                f.seek(pos)
                # read some small chunk
                st = f.read(2048)
                prev = pos
            sha.update(st)
    return sha.hexdigest()


def compute_long_sha(uri, buff_size=65536):
    """ Computes sha for a given uri
    :param uri: URI to compute fast_sha on
    :type uri: str
    :param buff_size: How much data to read at once
    :type buff_size: int
    :return sha: hexdigested sha
    :rtype: str
    """
    sha = hashlib.sha256()

    with open(uri, "rb") as f:
        while True:
            st = f.read(buff_size)
            if not st:
                break
                sha.update(st)
    return sha.hexdigest()


def create_new_db(name, engine='sina', db='sql',
                  token="", keyspace=None, cluster=None):
    """create_new_db creates a new Kosh database, adds a single user

    :param name: name of database
    :type name: str
    :param engine: engine to use, defaults to 'sina'
    :type engine: str, optional
    :param db: type of database for engine, defaults to 'sql', can be 'cass'
    :type db: str, optional
    :param token: for cassandra connection, token to use, defaults to "" means try to retrieve from user home dir
    :type token: str, optional
    :param keyspace: for cassandra keyspace to use, defaults to None means [user]_k
    :type keyspace: str, optional
    :param cluster: list of Casandra clusters to use
    :type cluster: list of str
    :return store: An handle to the Kosh store created
    :rtype: KoshStoreClass
    """
    user = os.environ["USER"]
    if db == 'sql' and name[-4:].lower() != ".sql":
        name += ".sql"
    if engine == "sina":
        cmd = "{}/init_sina.py --user={} --sina={} --sina_db={}".format(
            sys.prefix + "/bin",
            user,
            db,
            name)
    elif engine == 'cassandra':
        if keyspace is None:
            keyspace = user + "_k"
        cmd = "{}/init_cassandra.py --user={} --token={}" \
            "--keyspace={} --tables_root={} --cluster={}".format(
                sys.prefix + "/bin",
                user,
                token,
                keyspace,
                db,
                cluster)
    p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE)
    o, e = p.communicate()
    if engine == "sina":
        return kosh.KoshStore(engine="sina", db_uri=name)


class KoshScriptWrapper(object):
    def __init__(self,
                 executable,
                 named_parameters={},
                 kosh_names_mapping={},
                 positional_parameters=OrderedDict()):
        """Wrapper for scripts.
        Will inspect passed object at call time to construct command line and run script with appropriate parameters

        :param executable: path to executable/script, e.g. "python myscript.py"
        :type executable: str
        :param named_parameters: dictionary of parameters that will be passed as --parameter
                                 with their default values, e.g to always use --param1="one" as
                                 default value pass: {"param1":"one"}
                                 To use the scripts default value pass {"param1":"use_default"} that will lead to
                                 the argument not being passed to command line.
                                 To simply let all all parametrs to be as default you can pass a list of the parameters
                                 e.g ["param1", "param2"] is equivalent to:
                                 {"param1":"use_default", "param2":"use_default"}
                                 Note that if you want a parameter's value to picked up from the kosh object you send it
                                 must be defined here.
                                 Single or many dashed parameters (such as -o, or ---output) must be passed
                                 with the dash(es),
                                 e.g {"-o":"use_default", "---output":"use_default"}.
                                 Note: double dashed parameters may be passed that way as well but it is not necessary.
        :type named_parameters: dict or list
        :param kosh_names_mapping: if the named_parameter needs to be mapped to another attribute name in the
                                   passed object at call time this will do the remap.
                                   The format is {"script_param":"kosh_object_param"} or
                                   {"script_param":evaluator_function}
                                   where evaluator will take the kosh_object as an input and will return
                                   the desired value. Mapping to single or many dashes must be made to the
                                   dashless name.
        :type kosh_names_mapping: dict
        :param positional_parameters: these are "unamed" arguments that will be added at the end of the command line.
                                     they will be mapped to the default passed here or updated via the object passed at
                                     call time
        :type positional_parameters: OrderedDict
        """
        self.executable = executable
        if isinstance(named_parameters, dict):
            self.named_parameters = named_parameters
        elif isinstance(named_parameters, (list, tuple)):
            self.named_parameters = {}
            for name in named_parameters:
                self.named_parameters[name] = "use_default"
        else:
            raise RuntimeError(
                "named_parameters must be dict or list of keywords")
        if not isinstance(positional_parameters, OrderedDict):
            raise ValueError("positional_parameters must be ordered dict")
        self.positional_parameters = positional_parameters
        self.kosh_names_mapping = kosh_names_mapping

    def run(self, kosh_object, call_communicate=True,
            **updated_named_parameters):
        """Given a kosh object uses it to construct the appropriate command line
        :param kosh_object: The object that will be used to get values for named parameters
                            This object will also be the input to any mapping function
                            in `kosh_names_mapping` (see bellow)
        :type kosh_object: any
        :param call_communicate: After creating the subprocess do we call communicate?
        :type call_communicate: bool
        :param updated_named_parameters: these keyword values will be used to
                                         override anything generated and be passed
                                         as is to the command line
        :return out: A tuple of the output and err streams from the commuicate call or
                     The Popen process created if call_communicate is False
        """
        # Ok let's obtain the defaults
        named_parameters = copy.copy(self.named_parameters)
        remapped = {}
        for name in list(named_parameters.keys()):
            if name[0] == "-":
                new_name = name[1:]
                while new_name[0] == "-":
                    new_name = new_name[1:]
                named_parameters[new_name] = named_parameters[name]
                remapped[new_name] = name
                del(named_parameters[name])
        # let's take care of the single dash ones
        positional_parameters = copy.copy(self.positional_parameters)

        # First the easy part, one to one match
        for name in named_parameters:
            named_parameters[name] = getattr(
                kosh_object, name, named_parameters[name])

        # Let's map kosh names to script names first
        for name in self.kosh_names_mapping:
            mapping = self.kosh_names_mapping[name]
            if isinstance(mapping, str):
                # No mapper just gert value from kosh_object
                value = getattr(kosh_object, mapping)
            else:
                value = mapping(kosh_object)
            if name in named_parameters:
                named_parameters[name] = value
            if name in positional_parameters:
                positional_parameters[name] = value

        # Now let's update parameters with the user input
        for name in updated_named_parameters:
            if name in named_parameters:
                named_parameters[name] = updated_named_parameters[name]
            if name in self.positional_parameters:
                positional_parameters[name] = updated_named_parameters[name]

        # Ok we are ready to construct the command line
        cmd = self.executable

        for name in named_parameters:
            if named_parameters[name] == "use_default":
                # Let script handle it via its default value
                continue
            cmd_name = remapped.get(name, "--{}".format(name))
            cmd += " {} {}".format(cmd_name, str(named_parameters[name]))

        # Let's not forget positional params
        # Let's remove undefined trailing optional args
        if len(positional_parameters) > 0:
            while next(reversed(positional_parameters.values())
                       ) == "use_default":
                positional_parameters.popitem(last=True)
        cmd += " {}".format(" ".join([str(positional_parameters[x])
                                      for x in positional_parameters]))

        self.constructed_command_line = cmd
        p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE)
        if call_communicate:
            return p.communicate()
        else:
            return p
    __call__ = run
