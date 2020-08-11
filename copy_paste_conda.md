# Installing Kosh via conda

## Minimalist version (with HDF5)

### Just kosh and hdf5 loaders

```bash
conda create -n kosh -c conda-forge h5py numpy "python>3" sqlalchemy six pip networkx
conda activate kosh
git clone https://github.com/LLNL/Sina
cd sina/python
pip install -e .
cd ../..
git clone https://github.com/LLNL/kosh
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

### If you want the scikit-learn-based transformers you will need

*scikit-learn*

```bash
conda install -n kosh -c conda-forge scikit-learn
```

### if you want to run the tests you will need

*pytest*

```bash
conda install -n kosh -c conda-forge pytest-xdist pytest-cov
```

### if you want to run the notebooks you will need

*jupyterlab tqdm ipywidgets nb_conda nb_conda_kernels*

```bash
conda install -n kosh -c conda-forge tqdm ipywidgets jupyterlab nb_conda nb_conda_kernels
python -m ipykernel install --user --name kosh --display-name "Kosh Environment"
```

## Buffed up version with more loaders and extra but useful packages

This will let you build the documentation

```bash
conda create -n kosh -c conda-forge h5py numpy "python>3" sqlalchemy six ipython pip networkx sphinx pytest-xdist pytest-cov sphinx-autoapi pyflame jupyterlab flake8 autopep8 pillow coverage nbsphinx recommonmark tqdm ipywidgets nb_conda nb_conda_kernels scikit-learn
conda activate kosh
git clone https://github.com/LLNL/sina
cd sina/python
pip install -e .
cd ../..
git clone https://github.com/LLNL/kosh
cd kosh
python setup.py install
```
