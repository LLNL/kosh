from subprocess import Popen, PIPE
import shlex
import copy


class KoshScriptWrapper(object):
    def __init__(self,
                 executable,
                 ):
        """Wrapper for scripts.
        Will inspect passed object at call time to construct command line and run script with appropriate parameters

        :param executable: path to executable/script, e.g. "python myscript.py"
        :type executable: str
        """
        self.executable = executable
        self.arguments = []

    def add_argument(self, parameter, feed_attribute=None,
                     default="use_default", mapper=getattr, feed_pos=None):
        """Adds an argument to map
        :param parameter: name of the parameter in the script (e.g --file or -f)
                          use None to reflect positional argument
        :type parameter: str or none
        :param feed_attribute: name of attribute to use on the feed object
        :type feed_attribute: str
        :param default: default value to use if not affected by feed objects
                        to let the store use its own default pass "use_default"
        :param mapper: mapper to use defaults to gettattr,
                       the function will receive the kosh_object and
                       the kosh feed_attribute as arguments
        :type feed_attribute: function
        :param feed_pos: index of the feed object fed to use to do the mapping
                         -1 means as long as a feed object matches it will be used
                         and will override any value already set.
                         None means as soon as a feed object match that name
                         it will be used and the search is over.
        :type feed_pos: int or None
        """
        prefix = ""
        while len(parameter) > 0 and parameter[0] in ["-", "+"]:
            prefix += parameter[0]
            parameter = parameter[1:]
        if feed_attribute is None:
            feed_attribute = parameter
        self.arguments.append(
            {"parameter": parameter, "feed": feed_attribute, "mapper": mapper,
             "pos": feed_pos, "default": default,
             "prefix": prefix})

    def run(self,
            *kosh_objects,
            **updated_named_parameters):
        """Given a kosh object uses it to construct the appropriate command line
        :param kosh_object: The object that will be used to get values for named parameters
                            This object will also be the feed to any mapping function
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
        call_communicate = updated_named_parameters.get(
            "call_communicate", True)
        # Ok let's obtain the defaults
        named_parameters = {}
        stop_searching = []
        for i, kosh_object in enumerate(kosh_objects):
            # Let's find the arguments that can be extracted for this object
            for arg in self.arguments:
                param = arg["parameter"]
                inp = arg["feed"]
                if arg["pos"] == i or arg["pos"] == -1 or \
                        (arg["pos"] is None and param not in stop_searching and inp not in stop_searching):
                    if param not in named_parameters and param != "":
                        named_parameters[param] = arg["default"]
                    elif inp not in named_parameters and param == "":
                        named_parameters[inp] = arg["default"]

                    # Let's see if our object can construct a value
                    try:
                        value = arg["mapper"](kosh_object, inp)
                        if param != "":
                            named_parameters[param] = value
                        else:
                            named_parameters[inp] = value
                        if arg["pos"] is None:
                            if param == "":
                                stop_searching.append(inp)
                            else:
                                stop_searching.append(param)
                    except Exception:
                        pass

        # Now let's update parameters with the user feed
        for name in updated_named_parameters:
            if name in named_parameters:
                named_parameters[name] = updated_named_parameters[name]
        # Ok we are ready to construct the command line
        cmd = self.executable

        for arg in self.arguments:
            name = arg["parameter"]
            if name == "" or name not in named_parameters or named_parameters[
                    name] == "use_default":
                # Let script handle it via its default value
                continue
            cmd += " {}{} {}".format(arg["prefix"],
                                     name, str(named_parameters[name]))

        # Let's not forget positional params
        pos_values = []
        for arg in self.arguments:
            if arg["parameter"] == "":
                pos_values.append(named_parameters[arg["feed"]])
        # Let's remove undefined trailing optional arg
        # The have not be defined whatsoever
        while len(pos_values) > 0 and copy.copy(
                pos_values)[-1] == "use_default":
            pos_values.pop(-1)

        if len(pos_values) > 0:
            cmd += " {}".format(" ".join([str(x) for x in pos_values]))

        self.constructed_command_line = cmd
        p = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE)
        if call_communicate:
            return p.communicate()
        else:
            return p

    __call__ = run
