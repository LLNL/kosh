# Setup script for dkosh repo
from setuptools import setup, find_packages
from subprocess import Popen, PIPE
import shutil
import os


exec(open("./kosh/current_version.py").read())
version = current_version  # noqa
sha = None

git_describe_process = Popen(
    ("git",
     "describe",
     "--tags"),
    stdout=PIPE,
    stderr=PIPE)
if os.path.isdir(os.path.join(os.path.dirname(
    os.path.abspath(__file__)),'.git')):
    try:
        out, _ = git_describe_process.communicate()
        trial_version = out.decode("utf-8")
        sp = trial_version.split("-")
        trial_version = sp[0]
        # Clean tag?
        if len(sp) != 0:
            commits = sp[1]
            sha = sp[2]
            trial_version += "."+commits
        else:
            sha = None
        version = trial_version
    except Exception:
        pass

description="Manages Data Store with External Bulk Data"
if sha is not None:
    description += " (sha: {})".format(sha)

with open("README.md", "r") as fh:
    long_description = fh.read()
if os.path.exists("scripts/kosh"):
    os.remove("scripts/kosh")
shutil.copyfile("kosh/kosh_command.py", "scripts/kosh")
setup(name="kosh",
      version=version,
      description=description,
      url="https://github.com/LLNL/Kosh",
      author="Charles Doutriaux, Renee Olson",
      author_email="doutriaux1@llnl.gov",
      long_description=long_description,
      long_description_content_type="text/markdown",
      license="MIT",
      packages=find_packages(),
      scripts=["scripts/kosh",
               "scripts/kosh_command.py",
               "scripts/sbang",
               ],
      zip_safe=False,
      install_requires=[
          'llnl-sina >=1.14', 
          'networkx>=2.6',
          'numpy>=1.20',
          'scipy',
          'h5py>=3',
          'scikit-learn>=1.0.2',
          'pandas',
          'hdbscan',
          'matplotlib',
          'tqdm'
      ],
      classifiers=[
          "Programming Language :: Python",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ],
      )

