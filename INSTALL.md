# Installing Kosh

## Easy/No reading

If you do not want to read all this and simply copy/paste use the pages bellow
### conda

Copy/paste from [here](copy_paste_conda.md)

### pip

Copy/paste from [here](copy_paste_pip.md)

## Pre-requisites

### Some environement

#### Conda

```
conda create -n kosh -c conda-forge "python<3.8" networkx
```

#### Virtual Envirnoment

```
python -m virtualenv kosh
```

### Sina (LLNL's)

We use Sina to manage/store metadata, make sure you look at the casandra section if you plan on using cassandra.

Sina is on LLNL's github [here](https://github.com/LLNL/Sina)

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

We recommend adding the following packages matplotlib, h5py, scikit-learn, jupyter-lab and tqdm. pytest if you plan to run the test suite

#### Conda

```
conda install -n kosh -c conda-forge h5py jupyterlab nb_conda_kernels tqdm pytorch torchvision matplotlib pillow scikit-learn
```

```
conda install -n kosh -c conda-forge pytest-xdist pytest-cov
```

#### Virtual Environment

```
pip install h5py jupyterlab tqdm matplotlib pillow scikit-learn
```

```
pip install pytest-xdist pytest-cov
```

### Jupyter

Register your Python env via Jupyter:

```bash
python -m ipykernel install --user --name sonar-custom --display-name "Kosh Environment"
```
