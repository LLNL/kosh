# Installing Kosh via conda

## Minimalist version (with HDF5)

### Just kosh and hdf5 loaders

```bash
conda create -n kosh -c conda-forge h5py numpy "python>3" sqlalchemy six pip
git clone https://lc.llnl.gov/bitbucket/projects/SIBO/repos/sina/browse
cd sina/python
pip install -e .
cd ../..
git clone https://lc.llnl.gov/bitbucket/scm/ascaml/kosh.git
cd kosh
python setup.py install
```

### If you want the image loader you will need

*PIL*

```bash
conda install -n kosh -c conda-forge pillow
```

### If you want to build the documentation you will need

*sphinx sphinx-autoapi nbsphinx recommonmark*

```bash
conda install -n kosh -c conda-forge sphinx sphinx-autoapi nbsphinx recommonmark
```

### if you want to run the tests you will need

*nosetests*

```bash
conda install -n kosh -c conda-forge nose
```

### if you want to run the notebooks you will need

*jupyterlab tqdm ipywidgets*

```bash
conda install -n kosh -c conda-forge tqdm ipywidgets jupyterlab
```

## Buffed up version with more loaders and extra but useful packages

This will let you build the documentation

```bash
conda create -n kosh -c conda-forge h5py numpy "python>3" sqlalchemy six ipython pip sphinx nose sphinx-autoapi pyflame jupyterlab flake8 autopep8 pillow coverage nbsphinx recommonmark tqdm ipywidgets
git clone https://lc.llnl.gov/bitbucket/projects/SIBO/repos/sina/browse
cd sina/python
pip install -e .
cd ../..
git clone https://lc.llnl.gov/bitbucket/scm/ascaml/kosh.git
cd kosh
python setup.py install
```
