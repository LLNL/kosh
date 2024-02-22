# Installing Kosh via pip

## Minimalist version

### Just kosh and hdf5 loaders

```bash
export KOSH_PATH=kosh
pip install virtualenv  # just in case
python3 -m virtualenv $KOSH_PATH   # `kosh` can be any name/directory you want
source ${KOSH_PATH}/bin/activate
pip install h5py numpy sqlalchemy six pip networkx llnl-sina
git clone https://github.com/LLNL/kosh
cd kosh
pip install .
```

### If you want the image loader you will need

*PIL*

```bash
pip install pillow
```

### If you want to build the documentation you will need

*sphinx sphinx-autoapi nbsphinx recommonmark*

```bash
pip install sphinx sphinx-autoapi nbsphinx recommonmark
```

### If you want the scikit-learn-based transformers you will need

*scikit-learn*

```bash
pip install scikit-learn
```

### if you want to run the tests you will need

*pytest*

```bash
pip install pytest-xdist pytest-cov
```

### if you want to run the notebooks you will need

*jupyterlab tqdm ipywidgets*

```bash
pip install tqdm ipywidgets jupyterlab
python -m ipykernel install --user --name kosh --display-name "Kosh Environment"
```

## Buffed up version with more loaders and extra but useful packages

This will let you build the documentation

```bash
export KOSH_PATH=kosh
pip install virtualenv  # just in case
python3 -m virtualenv $KOSH_PATH   # `kosh` can be any name/directory you want
source ${KOSH_PATH}/bin/activate
pip install h5py numpy sqlalchemy six ipython pip networkx sphinx pytest-xdist pytest-cov pyflame jupyterlab flake8 autopep8 pillow coverage tqdm ipywidgets scikit-learn llnl-sina mkdocs mkdocstrings-python  mkdocs-jupyter mkdocs-material-extensions mkdocs-material mkdocs-literate-nav mkdocs-glightbox mkdocs-mermaid2-plugin mkdocs-gen-files mkdocs-material 
git clone https://github.com/LLNL/kosh
cd kosh
pip install .
```
