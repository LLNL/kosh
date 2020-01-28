# Setup script for dkosh repo
from setuptools import setup, find_packages

setup(name="kosh",
      version=0.1,
      description="Machine Learning Data Store",
      url="https://lc.llnl.gov/bitbucket/projects/ASCAML/repos/kosh/browse",
      author="Charles Doutriaux",
      author_email="doutriaux1@llnl.gov",
      license="MIT",
      packages=["kosh",],
      scripts=["scripts/init_cassandra.py",
               "scripts/init_sina.py",
               ],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False,
      install_requires=[
          'sina', 
      ],
      )
