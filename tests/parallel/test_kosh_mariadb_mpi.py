from __future__ import print_function
# from kosh.store import KoshStore
import os
import random
from mpi4py import MPI
import pytest
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  # noqa
from koshbase import KoshTest  # noqa


class KoshTestStore(KoshTest):
    @pytest.mark.mpi(min_size=2)
    def test_mariadb_mpi(self):

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        print("RANK:", rank, size, self.mariadb, file=sys.stderr)
        try:
            store, _ = self.connect(
                self.mariadb, execution_options={
                    "isolation_level": "READ UNCOMMITTED"})

        except Exception as err:
            print("Could not open store on rank:", rank, err, file=sys.stderr)
            comm.Abort()
        finally:
            # make sure count is created everywhere before moving on to scatter
            comm.barrier()
            print("passed br", rank, file=sys.stderr)

        try:
            if rank == 0:
                #
                # NOTE: I try to remove previous ensembles here, but the other ranks
                # can still sometimes have access to the ensembles being deleted here!
                # You can see this because the "count" attribute doesn't always match
                # rank 0.
                #
                ens = list(store.find_ensembles(name='bug-test'))
                for e in ens:
                    store.delete(e)

                metadata = {
                    'count': 0,
                }

                ens = store.create_ensemble(name='bug-test', metadata=metadata)

                #
                # NOTE: use a random number here to make sure that all ranks
                # are truly accessing the same ensemble (they aren't always!).
                #
                ens.count = random.randint(1, 10000)
                count = ens.count
            #
            # Wait until rank 0 finishes with the store.
            #
        except Exception as err:
            print(f"Rank 0 failed to prep the store with error {err}")
            comm.Abort()
        finally:
            comm.barrier()
        counts = list(store.find_ensembles(name='bug-test'))[0].count
        print(f"{rank}: {counts}", file=sys.stderr)
        gathered_count = comm.gather(counts, root=0)
        if rank == 0:
            print("RK 0:", gathered_count, count)
            self.assertEqual(gathered_count, [count, ] * size)


if __name__ == "__main__":
    A = KoshTestStore()
    for nm in dir(A):
        if nm[:4] == "test":
            fn = getattr(A, nm)
            print(nm, fn)
            fn()
