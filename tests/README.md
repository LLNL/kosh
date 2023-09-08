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

Name the file ".my.kosh.testdb.cnf"

## Parallel and Serial Tests
Some tests require more than one processor to run. 

To run parallel tets you need at least 4 processors. 
~~~
srun -n4 -p pdebug pytest --with-mpi tests/clusters/test_kosh_cluster*.py
~~~

The rest of the tests can be run with this command:
~~~
pytest tests/test_kosh*py
~~~