import kosh
import numpy as np
import h5py
from mpi4py import MPI
print("finished imports")


# MPI Communication with Kosh data management
# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()
print("Set up comm")

# This example shows ways to read in and operate on datasets that are too large to fit in memory.


# All ranks need this information
h5_file = "my_data.h5"
dataset_name = "normal"
data = np.empty((0, 5))

# Create large dataset on rank 0
if rank == 0:
    # Create an empty array with 5386 rows and 5 columns
    data = np.zeros((53786, 5))
    # Generate random normal values for each column, setting location (mean) to the column index
    # Column 0 will have close to a mean of zero, column 1 will have close to a mean of 1, etc.
    for i in range(5):
        data[:, i] = np.random.normal(loc=i, scale=1, size=53786)
    print(f"Array size: {data.shape}")

    # Save to hdf5 file
    with h5py.File(h5_file, "w") as f:
        f.create_dataset(dataset_name, data=data)
    print("Done creating h5 file")

comm.Barrier()
# We can store and organize all our datasets in a Kosh store
store_path = "data_slicing.sql"
store = kosh.connect(store_path, read_only=True)
print(f"Rank {rank} Created store: {store}")
dset  = store.create()

# Associate file to Kosh dataset
dset.associate(h5_file, 'hdf5')
print(f"Added data to Kosh dataset: {dset}")

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

# Get the total size of the dataset
total_size = next(dset[dataset_name].describe_entries())["size"][0]
print(f"Total data size: {total_size}")

start_index, end_index = get_slice(rank, nprocs, total_size)

# Read the local portion of the dataset
local_data = dset[dataset_name][slice(start_index, end_index)]

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

# Let's try using this loader with the getitem functionality.

# Create txt file using the same dataset as in the previous example
txt_name = 'array.out'
if rank == 0:
    txt_data = np.savetxt(txt_name, data, delimiter=',')

# We will use the same store as before, and add another Kosh dataset
dset2  = store.create()

# Associate the files to the Kosh dataset with array size in the metadata
data_shape = comm.bcast(data.shape, root=0)
metadata = {'size': data_shape[0]}
dset2.associate(txt_name, mime_type='numpy/txt', metadata=metadata)

# Get the total size of the dataset
total_size = getattr(dset2, "size")[0]
print(f"Total data size: {total_size}")

start_index, end_index = get_slice(rank, nprocs, total_size)

# Read the local portion of the dataset
local_data = dset2["features"][slice(start_index, end_index)]

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



# 3. Using a Kosh operator for MPI functions with parallel enabled loader

# In a Kosh operator, the __getitem_propogate__ function will propogate the required
# indices to the loader's __getitem__ function.

# __getitem_propogate__ needs to also receive the index of the input to which we will
# propogate the corresponding key.

# Let's continue our example with the default HDF5 loader, but this time the MPI functions will
# take place in a Kosh operator. The Kosh operator allows for multiple input files. 

# We will create a few arrays with varying row sizes, but all with 5 columns. Only do the work
# on rank 0 so we don't create duplicate files.
if rank == 0:
    for n in range(3):
        size = (n + 1) * 60
        # Create an empty array with 5386 rows and 5 columns
        data = np.zeros((size, 5))

        # Generate random normal values for each column, setting location (mean) to the column index
        # Column 0 will have close to a mean of zero, column 1 will have close to a mean of 1, etc.
        for i in range(5):
            data[:, i] = np.random.normal(loc=i, scale=1, size=size)

        # Save to hdf5 file
        h5_file = f"my_data{n}.h5"
        dataset_name = f"normal{n}"
        with h5py.File(h5_file, "w") as f:
            f.create_dataset(dataset_name, data=data)

# We will use the same store as before, and add another Kosh dataset
dset3  = store.create()

# Associate the files to the Kosh dataset
dset3.associate(["my_data0.h5", "my_data1.h5", "my_data2.h5"], 'hdf5')

# We need a function to assign data to each processor from multiple files

def distribute_data(total_size, nprocs, rank):

    # Calculate the start and end indices for each process
    start_idx = rank * total_size // nprocs
    end_idx = (rank + 1) * total_size // nprocs

    # Create a list to hold the local data sizes and corresponding file indices
    local_data_info = []
    current_size = 0

    for i, size in enumerate(sizes):
        if current_size + size > start_idx and current_size < end_idx:
            # Calculate the number of rows to read for this dataset
            if current_size < start_idx:
                # Calculate the starting row for this dataset
                start_row = start_idx - current_size
            else:
                start_row = 0
            
            if current_size + size > end_idx:
                # Calculate the number of rows to read
                end_row = end_idx - current_size
            else:
                end_row = size
            
            local_data_info.append((i, start_row, end_row))  # Store the index and row range
        current_size += size

    return local_data_info

# Now we can create a custom Kosh operator that can distribute data evenly across the
# processors and return the min and max of all the columns.

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

        # Get the sizes of each kosh dataset
        input_sizes = []
        desc = list(self.describe_entries())
        for i in range(len(inputs)):
            input_sizes.append(desc[i]["size"][0])

        total_size = sum(input_sizes)

        local_data_info = distribute_data(total_size, nprocs, rank)

        # Each process can now read its assigned datasets
        local_data = np.empty((0, 5), dtype=float)
        for index, start, stop in local_data_info:
            data = dset3[f"normal{index}"]
            local_data = np.concatenate(local_data, data, axis=0)

        # With MPI we calculate statistics for each column in the dataset
        global_min, global_max, _ = get_global_stats(local_data,
                                                     total_size,
                                                     comm)

        # Using the min and max we normalize each column of data
        for f in range(len(global_min)):
            local_data[:, f] = (local_data[:, f] - global_min[f]) / \
                (global_max[f] - global_min[f])

        return local_data

    def __getitem_propogate__(self, key, input_index):

        start = key.start
        stop = key.stop

        return slice(start, stop, key.step)

MPIN = MPINormalize(kosh_dset[feature_name])

sliced_data = MPIN[:, 2]
print(sliced_data)
        

# WARNING: You must be aware when creating custom loaders that you might not get
# the result you are expecting. See the Advanced Data Slicing example
