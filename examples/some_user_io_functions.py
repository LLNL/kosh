import numpy

def load_variable_from_file(filepath, variable_names):
    """Load the variable 'variable_name' for a file at filepath
    :param filepath: path to the file to read
    :type filepath: str
    :param variable_names: Name of the variable(s) to read in file
    :type variable_names: str or list
    :return: A numpy array containing the variable(s) values
    :rtype: numpy.ndarray
    """
    if not isinstance(variable_names, (list,tuple)):  # only one variable requested
        variable_names = [variable_names,]
    variables = [[],] * len(variable_names)
    previous_line = ""
    var_names = None
    with open(filepath, "rb") as f:
        for line in f.readlines():
            # Skip headers
            line = line.decode("utf-8")
            if line[0]=="#" or line.strip()=="end":
                previous_line = line
                var_names = None
                continue
            if var_names is None:
                var_names = previous_line.split()[1:]
                # clean up name list
                while "vs" in var_names:
                    var_names.remove("vs")
            sp = line.split()
            for ivar, name in enumerate(variable_names):
                if name in var_names:
                    index = var_names.index(name)
                    variables[ivar].append(float(sp[index]))
    # we're done reading these variables, co
    for ivar in range(len(variables)):
        if len(variables[ivar]) > 0 and isinstance(variables[ivar], list):
            variables[ivar] = numpy.array(variables[ivar])
    if len(variable_names) > 1:
        return variable_names
    else:  # only one variable read in
        return variables[0]
        
def get_variable_names(filepath):
    """Given a filename retrieves list of all vriables in the file
    :param filepath: Path to file
    :type filepath: str
    :return: list of variable names
    :rtype: list
    """
    variables = set()
    with open(filepath, "r") as f:
        previous = ""
        for line in f.readlines():
            if line[0] == "#":
                previous = line
                var_names = None
                continue
            if var_names is not None:
                continue
            var_names = previous.split()[1:]
            while "vs" in var_names:
                var_names.remove("vs")
            for name in var_names:
                variables.add(name)
    return list(variables)
