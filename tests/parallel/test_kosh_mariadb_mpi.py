from __future__ import print_function
# from kosh.store import KoshStore
import os
import random
import uuid
from mpi4py import MPI
import pytest
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  # noqa
from koshbase import KoshTest  # noqa


class KoshTestStore(KoshTest):
    @pytest.mark.mpi(min_size=2)
    def test_mariadb_mpi(self):
        """Ensure all MPI ranks observe the same MariaDB-backed ensemble state."""

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        ensemble_name = comm.bcast(
            f"bug-test-{uuid.uuid4().hex}" if rank == 0 else None,
            root=0,
        )
        ensemble_id = comm.bcast(
            f"bug-test-id-{uuid.uuid4().hex}" if rank == 0 else None,
            root=0,
        )

        print("RANK:", rank, size, self.mariadb, file=sys.stderr)
        store = None
        connect_error = None
        try:
            store, _ = self.connect(
                self.mariadb, execution_options={
                    "isolation_level": "READ COMMITTED"})
        except Exception as err:
            connect_error = f"Could not open store on rank {rank}: {err}"

        connect_errors = comm.allgather(connect_error)
        self.assertEqual(
            connect_errors,
            [None] * size,
            msg="\n".join(error for error in connect_errors if error is not None),
        )
        print("passed connect", rank, file=sys.stderr)

        count = None
        prep_error = None
        try:
            if rank == 0:
                metadata = {'count': 0}
                ens = store.create_ensemble(name=ensemble_name, id=ensemble_id, metadata=metadata)
                ens.count = random.randint(1, 10000)
                count = ens.count
        except Exception as err:
            prep_error = f"Rank {rank} failed to prepare/query the store: {err}"

        prep_errors = comm.allgather(prep_error)
        self.assertEqual(
            prep_errors,
            [None] * size,
            msg="\n".join(error for error in prep_errors if error is not None),
        )

        count = comm.bcast(count, root=0)

        matches = list(store.find_ensembles(name=ensemble_name))
        query_error = None
        payload = None
        if len(matches) != 1:
            query_error = (
                f"Rank {rank} expected exactly one ensemble named {ensemble_name!r}, "
                f"found {len(matches)}"
            )
        else:
            payload = (matches[0].id, matches[0].count)
            print(f"{rank}: {payload[1]}", file=sys.stderr)

        gathered = comm.gather((query_error, payload), root=0)
        if rank == 0:
            query_errors = [error for error, _ in gathered if error is not None]
            self.assertEqual(query_errors, [], msg="\n".join(query_errors))
            gathered_ids = [payload[0] for _, payload in gathered]
            gathered_counts = [payload[1] for _, payload in gathered]
            self.assertEqual(gathered_ids, [ensemble_id] * size)
            self.assertEqual(gathered_counts, [count] * size)

        store.close()


if __name__ == "__main__":
    A = KoshTestStore()
    for nm in dir(A):
        if nm[:4] == "test":
            fn = getattr(A, nm)
            print(nm, fn)
            fn()
