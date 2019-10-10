# Setup script for dkosh repo
from setuptools import setup, find_packages

setup(name="kosh",
      version=0.1,
      description="Machine Learning Data Store",
      url="https://lc.llnl.gov/bitbucket/projects/ASCAML/repos/kosh/browse",
      packages=find_packages(),
      package_dir={'kosh': 'kosh'},
      scripts=["scripts/init_cassandra.py",
          "scripts/init_sina.py",
          ],
)
