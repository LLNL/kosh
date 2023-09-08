# Welcome to the Kosh test suite

## To run the Kosh CI follow these steps:

## Create the virtual environment
Create and activate the virtual environment. It must have this name and be located in a directory named gitlab in your workspace.
~~~
/usr/tce/bin/python3 -m venv /usr/WS1/<username>/gitlab/gitlab_kosh_py3
~~~
~~~
source /usr/WS1/<username>/gitlab/gitlab_kosh_py3/bin/activate
~~~

## Install necessary libraries
~~~
pip install --upgrade pip pytest pytest-cov pytest-xdist pytest-mpi mpi4py flake8 mysql-connector-python
~~~
~~~
pip install -e .
~~~

## Create a SSH key
Create a new SSH RSA key titled "gitlab"
https://hpc.llnl.gov/cloud/services/GitLab/create-ssh-keys/

## MariaDB Server Tests
There are two tests that require connection to a MariaDB server. You can request the file from a Kosh developer. doutriaux1@llnl.gov, muryanto1@llnl.gov, moreno45@llnl.gov, olson59@llnl.gov

Save the file as ".my.kosh.testdb.cnf" in your home directory.

## Run test locally on all nodes
Repeatedly ssh into all the nodes possible. For example, "Runner: #254 (f42dVtUo_) ruby967-shell"

On each node run:
~~~
pytest
~~~
There will be a prompt that asks for connection and type yes

## Running parallel and serial tests locally
Some tests will fail if not run with adequate resources.
~~~
srun -n4 -p pdebug pytest --with-mpi tests/clusters/test_kosh_cluster*.py
~~~

The rest of the tests can be run with these commands:
~~~
pytest tests/clusters/test_kosh_cluster*.py
~~~
~~~
pytest -s --cov=kosh tests/non_parallel/test_kosh_*.py
~~~
~~~
pytest --cov=kosh --cov-append -n 16 tests/test_kosh_*.py
~~~


