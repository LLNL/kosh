# Installing Kosh

## Pre-requisites

### Some environement

#### LC provided virtual env

```
source /collab/usr/gapps/wf/releases/sina/bin/activate
```

#### Conda

```
conda create -n Kosh -c conda-forge "python<3.8"
```

#### Virtual Envirnoment

```
python -m virtualenv kosh
```

### Sina

We use Sina to manage/store metadata, make sure you look at the casandra section i you plan on using cassandra.


Sina. For details see [Sina's Readme](https://lc.llnl.gov/workflow/docs/sina/readme.html)


#### Conda

```
conda install -n kosh -c conda-forge tox flake8 mock jsonschema SQLAlchemy
```

For cassandra also add:

```
conda install -n kosh -c conda-forge cython cassandra-driver
```

#### pip


```
pip install tox flake8 mock jsonschema SQLAlchemy
```

For cassandra also add:

```
pip install cython cassandra-driver
```


### Niceties

We recommend adding the following packages h5py, jupyter-lab and tqdm. Nosetests if you plan to run the test suite

#### Conda

```
conda install -n kosh -c conda-forge h5py jupyterlab nb_conda_kernels nbtqdm
```

```
conda install -n kosh -c conda-forge nose
```

#### Virtual Envirnoment

```
pip install h5py jupyterlab tqdm
```

```
pip install nose
```

