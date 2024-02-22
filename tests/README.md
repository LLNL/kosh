# Welcome to the Kosh test suite

## To run the tests locally follow these steps:

## Requirements for Testing
There are a few modules needed for testing: run 
~~~
pip install --upgrade pip pytest pytest-cov pytest-xdist pytest-mpi mpi4py flake8 mysql-connector-python
~~~

## MariaDB Server Tests
There are two tests that require connection to a MariaDB server: test_loaders_mariadb and test_disable_lock_file. If you want these tests to pass when run locally you can get a free download.

https://mariadb.com/downloads/community/

The MariaDB URI can be controlled via env variable KOSH_TEST_MARIADB and the location to the cnf file is controlled via the env variable KOSH_TEST_MARIACNF

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