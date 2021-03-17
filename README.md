![Kosh Logo](share/icons/png/Kosh_Logo_Blue.png)
# Overview
Kosh allows codes to store, query, share data via an easy-to-use Python API. Kosh lies on top of Sina and as a result can use any database backend supported by Sina.

In addition Kosh aims to make data access and sharing as simple as possible.

Via "loaders" Kosh can open files associated with datasets in a seamless fashion independently of the actual file format. Kosh's loader can also load data in different format, although numpy is the most usual output type. 



# Getting Started


To get the latest public version:

```bash
pip install kosh
```

To get the latest stable, from a cloned repo simply run:

```bash
pip install .
```

Alternatively  add the path to this repo to your `PYTHONPATH` environment variable, or in your code with:

```python
import sys
sys.path.append(path_to_kosh_repo)
```

For more details look into the [installation doc](INSTALL.md)

# First steps

See [this file](docs/source/users/index.md)

# Getting Involved
Kosh is user-oriented, and users' questions, comments, and contributions help guide its evolution. We welcome involvement and feedbacks.

# Contact Info
You can reach our team at kosh-support@llnl.gov.
Kosh main developer can be reached at: doutriaux1@llnl.gov

# Contributing
Contributions should be submitted as a pull request pointing to the develop branch, and must pass Kosh's CI process; to run the same checks locally, use:
```
pytest tests/test_kosh*py
```

Contributions must be made under the same license as Kosh (see the bottom of this file).

# Release and License
Kosh is distributed under the terms of the MIT license; new contributions must be made under this license.

SPDX-License-Identifier: MIT

``LLNL-CODE-814755``
