#!/usr/bin/env python
import glob
import argparse
import kosh
import shlex
import os
from subprocess import Popen, PIPE
import sys


kosh_version = kosh.__version__

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    "--version",
    "-v",
    help="version to use in files",
    default=kosh_version)
parser.add_argument("--svg", "-s", help="svgs to convert")
parser.add_argument(
    "--no-sha",
    help="Will not write the version with SHA",
    action="store_true")

args = parser.parse_args()
if args.svg is None:
    svgs = glob.glob(os.path.join("share", "icons", "svg", "*_Text.svg"))
else:
    svgs = [args.svg, ]

version = args.version
if version[0] == "v":
    version = version[1:]
if ".g" in version:
    short_version = version.split(".g")[0]  # Removes the git extension
    short_version = ".".join(
        short_version.split(".")[
            :-1])  # Removes the git delta
else:
    short_version = version
if args.no_sha or version == short_version:
    version = ""

for svg in svgs:
    with open(svg) as f:
        content = f.read()
    content = content.replace("FULL_VERSION", version)
    content = content.replace("VERSION_NUMBER", short_version)
    svg_name = svg.split("_Text.svg")[0] + ".svg"
    with open(svg_name, "w") as f:
        f.write(content)
    png_name = svg_name.replace("svg", "png")
    cmd = "inkscape --file {} --export-png {} --export-width=1035".format(
        svg_name, png_name)
    if not sys.platform.startswith("win"):
        cmd = shlex.split(cmd)
    p = Popen(cmd, stdout=PIPE, stderr=PIPE)
    o, e = p.communicate()
