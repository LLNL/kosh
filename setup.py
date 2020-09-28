# Setup script for dkosh repo
from setuptools import setup, find_packages
from subprocess import Popen, PIPE

version = "1.0"
git_describe_process = Popen(
    ("git",
     "describe",
     "--tags"),
    stdout=PIPE,
    stderr=PIPE)
try:
    out, _ = git_describe_process.communicate()
    version = out.decode("utf-8").replace("-", ".")
except Exception:
    pass

try:
    # Let's make sure sina is installed
    import sina
except ImportError:
    raise RuntimeError("Please install Sina from: https://github.com/LLNL/Sina")

try:
    # Let's make sure the correct sina is installed
    import sina.datastores.sql as sina_sql
except ImportError:
    raise RuntimeError("You appear to have Sina installed but it seems to be another Sina."
                       "\nKosh requires LLNL's version of Sina"
                       "\nPlease install Sina from: https://github.com/LLNL/Sina")

setup(name="kosh",
      version=version,
      description="Machine Learning Data Store",
      url="https://github.com/LLNL/Kosh",
      author="Charles Doutriaux",
      author_email="doutriaux1@llnl.gov",
      license="MIT",
      packages=find_packages(),
      scripts=["scripts/init_sina.py",
               "scripts/kosh",
               ],
      zip_safe=False,
      install_requires=[
          'sina', 
      ],
      )
