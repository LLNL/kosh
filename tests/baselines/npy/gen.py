from __future__ import print_function
import numpy

name = "example1.txt"

n = 25
ncols = 5
with open(name, "w") as f:
    for i in range(n):
        print("{} ".format(i), file=f, end="")
        data = numpy.random.random(ncols) + numpy.arange(ncols)
        for j in range(ncols):
            print("{} ".format(data[j]), file=f, end="")
        print("\n", file=f, end="")
