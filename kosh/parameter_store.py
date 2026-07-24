from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import kosh
from sina.utils import DataRange


@dataclass(frozen=True)
class Association:
    path: str
    mime_type: str


@dataclass(frozen=True)
class StepRequest:
    """Declarative request for creating/updating datasets by parameter set."""

    store_uri: str
    step: str
    ensemble_name: Optional[str] = None
    init_params: Mapping[str, Any] = field(default_factory=dict)
    dataset_record_type: Optional[str] = None
    rtol: float = 1e-5
    atol: float = 1e-8
    strict_match: bool = False
    upsert_init: bool = False
    delete_ensemble: bool = False
    wipe_store: bool = False
    step_field: str = "workflow_step"
    init_step_value: str = "init"
    metadata_updates: Mapping[str, Any] = field(default_factory=dict)
    dataset_associations: Sequence[Association] = field(default_factory=tuple)
    ensemble_associations: Sequence[Association] = field(default_factory=tuple)
    ensemble_metadata_updates: Mapping[str, Any] = field(default_factory=dict)
    update_record: Optional[Mapping[str, Any]] = None
    connect_kwargs: Mapping[str, Any] = field(default_factory=dict)


def _data_range_for_value(value: float, *, rtol: float, atol: float) -> DataRange:
    lower = value - (atol + rtol * abs(value))
    upper = value + (atol + rtol * abs(value))
    if lower == upper:
        return DataRange(lower, upper, max_inclusive=True)
    return DataRange(lower, upper)


def _build_query(
    init_params: Mapping[str, Any], *, rtol: float, atol: float, strict_match: bool
) -> Dict[str, Any]:
    query: Dict[str, Any] = {}
    for key, value in init_params.items():
        if isinstance(value, (int, float)):
            numeric = float(value)
            if strict_match:
                query[key] = numeric
            else:
                query[key] = _data_range_for_value(numeric, rtol=rtol, atol=atol)
        else:
            query[key] = value
    return query


def _open_store(store_uri: str, *, wipe_store: bool, connect_kwargs: Mapping[str, Any]) -> kosh.KoshStore:
    if wipe_store:
        return kosh.connect(store_uri, delete_all_contents=True, **dict(connect_kwargs))
    return kosh.connect(store_uri, **dict(connect_kwargs))


def _get_or_create_ensemble(store: kosh.KoshStore, *, name: str, delete_ensemble: bool):
    ensembles = list(store.find_ensembles(name=name))
    if len(ensembles) > 1:
        raise RuntimeError(f"Found more than 1 ensemble matching name {name}")
    if len(ensembles) == 1:
        if delete_ensemble:
            store.delete(ensembles[0])
            return store.create_ensemble(name=name)
        return ensembles[0]
    return store.create_ensemble(name=name)


def apply_step(request: StepRequest) -> List[str]:
    """Create or update dataset(s) in a Kosh store for a workflow step.

    Behavior:
    - If `request.step == request.init_step_value`: create a dataset with `init_params` (and add to the
      ensemble when `request.ensemble_name` is provided).
    - Otherwise: find dataset(s) matching `init_params` (DataRange for numerics), then update. If
      `request.ensemble_name` is provided, restrict the search to that ensemble.
    """
    store = _open_store(request.store_uri, wipe_store=request.wipe_store, connect_kwargs=request.connect_kwargs)
    try:
        record_type = request.dataset_record_type
        find_kwargs: Dict[str, Any] = {}
        if record_type:
            find_kwargs["types"] = [record_type]

        ensemble = None
        if request.ensemble_name is None:
            if request.delete_ensemble:
                raise ValueError("delete_ensemble requires an ensemble_name")
            if request.ensemble_associations:
                raise ValueError("ensemble_associations requires an ensemble_name")
            if request.ensemble_metadata_updates:
                raise ValueError("ensemble_metadata_updates requires an ensemble_name")
        else:
            ensemble = _get_or_create_ensemble(
                store, name=request.ensemble_name, delete_ensemble=request.delete_ensemble
            )

            for assoc in request.ensemble_associations:
                ensemble.associate(assoc.path, assoc.mime_type)
            if request.ensemble_metadata_updates:
                ensemble.update(dict(request.ensemble_metadata_updates))

        if request.step == request.init_step_value:
            dataset_ids: List[str] = []
            if request.upsert_init:
                query = _build_query(
                    request.init_params,
                    rtol=request.rtol,
                    atol=request.atol,
                    strict_match=request.strict_match,
                )
                if ensemble is None:
                    dataset_ids = list(store.find(ids_only=True, **find_kwargs, **query))
                else:
                    dataset_ids = list(ensemble.find_datasets(ids_only=True, **query))
                if len(dataset_ids) > 1:
                    raise ValueError(f"Found more than one dataset {len(dataset_ids)} matching init params, giving up")
            if not dataset_ids:
                metadata = dict(request.init_params)
                metadata[request.step_field] = request.init_step_value
                dataset = store.create(metadata=metadata)
                if ensemble is not None:
                    ensemble.add(dataset)
                dataset_ids = [dataset.id]
        else:
            query = _build_query(
                request.init_params,
                rtol=request.rtol,
                atol=request.atol,
                strict_match=request.strict_match,
            )
            if ensemble is None:
                dataset_ids = list(store.find(ids_only=True, **find_kwargs, **query))
            else:
                dataset_ids = list(ensemble.find_datasets(ids_only=True, **query))

        if not dataset_ids:
            raise ValueError("could not find any dataset that match your request")

        for dataset_id in dataset_ids:
            ds = store.open(dataset_id)
            meta = dict(request.metadata_updates)
            meta[request.step_field] = request.step
            if request.update_record:
                meta.update(dict(request.update_record))
            if meta:
                ds.update(meta)
            for assoc in request.dataset_associations:
                ds.associate(assoc.path, assoc.mime_type)

        return dataset_ids
    finally:
        store.close()
