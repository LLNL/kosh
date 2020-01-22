# Installing Kosh

## Easy/No reading

If you do not want to read all this and simply copy/paste use the pages bellow
### conda

Copy/paste from [here](copy_paste_conda.md)

### pip

Copy/paste from [here](copy_paste_pip.md)

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

Sina is on LC's bitbucket [here](https://lc.llnl.gov/bitbucket/projects/SIBO/repos/sina/browse)

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

We recommend adding the following packages matplotlib, h5py, jupyter-lab and tqdm. Nosetests if you plan to run the test suite

#### Conda

```
conda install -n kosh -c conda-forge h5py jupyterlab nb_conda_kernels tqdm pytorch torchvision matplotlib pillow
```

```
conda install -n kosh -c conda-forge nose
```

#### Virtual Environment

```
pip install h5py jupyterlab tqdm matplotlib pillow
```

```
pip install nose
```

### Jupyter

Make sure you get your env registered on LC jupyter lab via:

```bash
python -m ipykernel install --user --name sonar-custom --display-name "Kosh Environment"
```
