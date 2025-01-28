import kosh
import numpy as np
import h5py
from mpi4py import MPI



# MPI Communication with Kosh data management


# This example shows ways to read in and operate on datasets that are too large to fit in memory.


# Create large dataset

# Create an empty array with 5386 rows and 5 columns
data = np.zeros((539786, 5))

# Generate random normal values for each column, setting location (mean) to the column index
# Column 0 will have close to a mean of zero, column 1 will have close to a mean of 1, etc.
for i in range(5):
    data[:, i] = np.random.normal(loc=i, scale=1, size=539786)

# Save to hdf5 file
h5_file = "my_data.h5"
dataset_name = "normal"
with h5py.File(h5_file, "w") as f:
    f.create_dataset(dataset_name, data=data)

# We can store and organize all our datasets in a Kosh store
store_path = "data_slicing.sql"
store = kosh.connect(store_path, delete_all_contents=True)
dset  = store.create()

# Associate file to Kosh dataset
dset.associate(h5_file, 'hdf5' )

# HDF5 already allows us to load slices of the data without reading in the entire dataset. 
# Kosh's Default HDF5 Loader allows us to do the same thing with Kosh datasets pointing to HDF5 files.


# Using MPI communication 3 ways with Kosh tools

# 1. MPI Function with default HDF5 loader

# We need to distribute the data across ranks or processors. Each process will run this function and since
# it is based on rank number, each one will read in a different chunk of data.

def get_slice(rank, nprocs, total_size):

    # Calculate the chunk size for each rank
    chunk_size = total_size // nprocs
    remainder = total_size % nprocs

    # Calculate start and end indices for each rank
    start_index = rank * chunk_size + min(rank, remainder)
    end_index = start_index + chunk_size + (1 if rank < remainder else 0)

    return [start_index, end_index]

# Next we want to use MPI communication to calculate statistics of each column of the dataset. We
# will calculate the minimum, maximum, and mean. 

def get_global_stats(local_data, total_size, comm):

    # Now we can process the local data
    local_min = np.min(local_data, axis=0)
    local_max = np.max(local_data, axis=0)
    local_sum = np.sum(local_data, axis=0)

    # Use MPI comm to find global stats
    global_min = local_min * 0.0
    comm.Allreduce([local_min, MPI.DOUBLE],
                   [global_min, MPI.DOUBLE],
                   op=MPI.MIN)

    global_max = local_max * 0.0
    comm.Allreduce([local_max, MPI.DOUBLE],
                   [global_max, MPI.DOUBLE],
                   op=MPI.MAX)

    global_sum = comm.allreduce(local_sum, op=MPI.SUM)
    global_mean = global_sum / total_size

    return [global_min, global_max, global_mean]

# Let's use these functions with our Kosh store and dataset

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

# Get the total size of the dataset
total_size = next(dset[dataset_name].describe_entries())["size"][0]
print(f"Total data size: {total_size}")

start_index, end_index = get_slice(rank, nprocs, total_size)

# Read the local portion of the dataset
local_data = kosh_dset[feature_name][slice(start_index, end_index)]

# Each process now has its own portion of the dataset
print(f"Rank {rank} has data size: {local_data.shape}")

global_min, global_max, global_mean = get_global_stats(local_data,
                                                       total_size,
                                                       comm)

# Each process was able to communicate the local statsistics, and process 0 did
# the final communication to compute the global statistics for each column.

print("Data stats: min, max, mean\n")
for i in range(len(global_min)):
    print(f"Column {i}: {global_min[i]}, {global_max[i]}, {global_mean[i]}")


# 2. MPI Function with a custom Kosh loader

# Kosh offers loaders for these types of data files.
#     HDF5
#     json
#     numpy
#     pandas
#     pgm
#     pil
#     sidre

# Only the default HDF5 and numpy text loaders have the ability to enable distributed data
# loading. 

# However, users may create custom loaders that may enable distributed data loading by adding
# a __getitem__ method. The numpy text loader is a good example of how to use __getitem__ in a
# custom loader. 

class NumpyTxtLoader(kosh.KoshLoader):
    types = {"numpy/txt": ["numpy", ]}

    def _setup_via_metadata(self):
        # use metadata to identify
        self.skiprows = getattr(self.obj, "skiprows", 0)
        self.features_at_line = getattr(self.obj, "features_line", None)
        self.features_separator = getattr(self.obj, "features_separator", None)
        self.columns_width = getattr(self.obj, "columns_width", None)

    def list_features(self, *args, **kargs):
        self._setup_via_metadata()
        if self.features_at_line is None:
            return ["features", ]
        else:
            with open(self.obj.uri) as f:
                line = -1
                while line < self.features_at_line:
                    st = f.readline()
                    line += 1
                if self.columns_width is not None:
                    features = [st[i:i + self.columns_width].strip()
                                for i in range(0, len(st), self.columns_width)]
                else:
                    while st[0] == "#":
                        st = st[1:]
                    st = st.strip()
                    features = st.split(self.features_separator)
            return features

    def extract(self):
        self._setup_via_metadata()
        return self[:]

    def __getitem__(self, key):
        self._setup_via_metadata()
        original_key = key
        if isinstance(key, tuple):
            # double subset
            key, key2 = key[:2]  # ignore if more is sent
        else:
            key2 = None
        if isinstance(key, int):
            key = slice(key, key + 1)

        if isinstance(key, slice):
            start = key.start
            stop = key.stop
            step = key.step
            if step is None:
                step = 1
            if (start is not None and start < 0) or \
                    (stop is not None and stop < 0):
                # Doh we need to count lines
                nlines = number_of_lines(self.obj.uri)
                if start is not None and start < 0:
                    start = nlines + start
                if stop is not None and stop < 0:
                    stop = nlines + stop
            # ok if it's neg step we need to flip these two
            # it has to do with numpy loader starting at a row for n row
            # not reading a range
            if step < 0:
                start, stop = stop, start
                if stop is not None:
                    stop += 1  # because slice if exclusive on the end
            if start is None:
                start = self.skiprows
            else:
                start += self.skiprows
            if stop is not None:
                max_rows = stop - start + self.skiprows
            else:
                max_rows = None

            if max_rows is None or max_rows > 0:
                # , usecols=numpy.arange(key2.start, key2.stop, key2.step))
                data = numpy.loadtxt(
                    self.obj.uri, skiprows=start, max_rows=max_rows)
            else:
                # , usecols=numpy.arange(key2.start, key2.stop, key2.step))
                data = numpy.loadtxt(
                    self.obj.uri,
                    skiprows=start,
                    max_rows=2)[
                    0:2:-1]
            if data.ndim > 1:
                if key2 is not None:
                    data = data[::step, key2]
                elif step != 1:  # useless if step is 1
                    data = data[::step]
            else:
                if key.step is not None and key.step != 1:
                    data = data[::step]
                else:
                    if key2 is not None:
                        data = data[key2]
                    else:
                        data = data
        else:
            raise KeyError("Invalid key value: {}".format(original_key))
        if self.features_at_line is None:
            return data
        else:
            feature_index = self.list_features().index(self.feature)
            return data[:, feature_index]

# 3. Using a Kosh operator for MPI functions with parallel enabled loader

# In a Kosh operator, the __getitem_propogate__ function will propogate the required
# indices to the loader's __getitem__ function.

# __getitem_propogate__ needs to also receive the index of the input to which we will
# propogate the corresponding key.

# Let's continue our example with an HDF5 data file but this time the MPI functions will
# take place in a Kosh operator, and using the global min and max we can return normalized
# dataset chunks to each process.

class MPINormalize(kosh.KoshOperator):

    types = {"numpy": ["numpy", ]}

    def __init__(self, *args, **options):
        super(KoshCluster, self).__init__(*args, **options)
        self.options = options

        # Initialize MPI
        self.comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()
        self.nprocs = comm.Get_size()

    def operate(self, *inputs, **kargs):

        # It's possible to give operators multiple Kosh datasets
        # This operator is assuming only one
        kosh_dset = inputs[0]

        # Get the total size of the dataset
        total_size = next(kosh_dset.describe_entries())["size"][0]

        # Each process will load the portion of the data assigned to it
        start_index, end_index = get_slice(rank, nprocs, total_size)

        # Read the local portion of the dataset
        local_data = kosh_dset[feature_name][slice(start_index, end_index)]

        # With MPI we calculate statistics for each column in the dataset
        global_min, global_max, _ = get_global_stats(local_data,
                                                     total_size,
                                                     comm)

        # Using the min and max we normalize each column of data
        for f in range(len(global_min)):
            data[:, f] = (data[:, f] - global_min[f]) / \
                (global_max[f] - global_min[f])

        return data

    def __getitem_propogate__(self, key, input_index):

        start = key.start
        stop = key.stop

        return slice(start, stop, key.step)

MPIN = MPINormalize(kosh_dset[feature_name])

sliced_data = MPIN[:, 2]
print(sliced_data)
        

# WARNING: You must be aware when creating custom loaders that you might not get
# the result you are expecting. See the Advanced Data Slicing example


















##########################################################################################

####################### Use this capability in a Kosh operator ###########################

##########################################################################################



class ParallelDataPrep(kosh.KoshOperator):
    types = {"range": ["numpy", ]}

    def __init__(self, *inputs, **kargs):
        super(ParallelDataPrep, self).__init__(*inputs, **kargs)

    def operate(self, *inputs, **args):

        # Initialize MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        nprocs = comm.Get_size()

    def __getitem__(self, key):
        if isinstance(key, slice):
            # Return a slice of the dataset
            return self.dataset[key]
        else:
            # Return a single item from the dataset
            return self.dataset[key]



# # Make sure our custom loader is picked up
# del store.loaders["hdf5"]
# store.add_loader(MySlicingLoader)
# loader = MySlicingLoader(store_path, dset[dataset_name])

# # Get the total size of the dataset
# total_size = loader.get_data_size()

# # Calculate the chunk size for each rank
# chunk_size = total_size // size
# remainder = total_size % size

# # Calculate start and end indices for each rank
# start_index = rank * chunk_size + min(rank, remainder)
# end_index = start_index + chunk_size + (1 if rank < remainder else 0)

