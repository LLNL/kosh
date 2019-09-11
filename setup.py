# Setup script for dkosh repo
from setuptools import setup, find_packages

setup(name="dkosh",
      version=0.1,
      description="Machine Learning Data Store",
      url="https://lc.llnl.gov/bitbucket/projects/ASCAML/repos/dkosh/browse",
      packages=find_packages(),
      package_dir={'dkosh': 'dkosh'},
      scripts=["scripts/init_cassandra.py"],
)