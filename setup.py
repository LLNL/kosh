# Setup script for dkosh repo
from setuptools import setup, find_packages

setup(name="kosh",
      version=1.0,
      description="Machine Learning Data Store",
      url="https://lc.llnl.gov/bitbucket/projects/ASCAML/repos/kosh/browse",
      author="Charles Doutriaux",
      author_email="doutriaux1@llnl.gov",
      license="MIT",
      packages=find_packages(),
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
