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

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name="kosh",
      version=version,
      description="Machine Learning Data Store",
      url="https://github.com/LLNL/Kosh",
      author="Charles Doutriaux",
      author_email="doutriaux1@llnl.gov",
      long_description=long_description,
      long_description_content_type="text/markdown",
      license="MIT",
      packages=find_packages(),
      scripts=["scripts/init_sina.py",
               "scripts/kosh",
               "scripts/sbang",
               ],
      zip_safe=False,
      install_requires=[
          'llnl-sina', 
          'networkx',
          'numpy',
      ],
      classifiers=[
          "Programming Language :: Python",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ],
      )
Popen(("scripts/render_logos.py",)).communicate()
