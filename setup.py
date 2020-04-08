# Setup script for dkosh repo
from setuptools import setup, find_packages
from subprocess import Popen, PIPE

version = "1.1"
p = Popen(
    ("git",
     "describe",
     "--tags"),
    stdout=PIPE,
    stderr=PIPE)
try:
    o, e = p.communicate()
    version = o.decode("utf-8").replace("-", ".")
    print("Verion:", version)
except Exception:
    pass

print("Version:" , version)

setup(name="kosh",
      version=version,
      description="Machine Learning Data Store",
      url="https://lc.llnl.gov/bitbucket/projects/ASCAML/repos/kosh/browse",
      author="Charles Doutriaux",
      author_email="doutriaux1@llnl.gov",
      license="MIT",
      packages=find_packages(),
      scripts=["scripts/init_cassandra.py",
               "scripts/init_sina.py",
               "scripts/kosh",
               ],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False,
      install_requires=[
          'sina', 
      ],
      )
