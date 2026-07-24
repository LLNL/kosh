from __future__ import print_function

import argparse
import ast
import sys
import warnings
from datetime import datetime
from difflib import get_close_matches

import kosh
from sina.utils import DataRange

from .parameter_store import Association, StepRequest, apply_step


def _str2bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got: {value!r}")


def _parse_kv_pairs(pairs):
    out = {}
    for term in pairs:
        if "=" not in term:
            raise ValueError("Parameters must be in form 'key=value'")
        key, value = term.split("=", 1)
        try:
            out[key] = ast.literal_eval(value)
        except Exception:
            out[key] = value
    return out


def _known_long_option_names(parser):
    return sorted(
        {
            option_string[2:]
            for action in parser._actions
            for option_string in action.option_strings
            if option_string.startswith("--")
        }
    )


def _dataset_type_filter_kwargs(dataset_record_type):
    if dataset_record_type:
        return {"types": [dataset_record_type]}
    return {}


def _collect_implicit_param_terms(parser, implicit_args, no_typo_check=False):
    """Convert unknown CLI tokens into workflow parameter terms.

    This allows users to pass simple workflow parameter filters as implicit
    flags such as ``--mach=0.8`` or ``--mach 0.8`` while preserving the
    explicit ``--param key=value`` form for parameter names that are not safe
    to encode as CLI option names. When typo checking is enabled, tokens that
    look like misspelled workflow options are rejected with a suggestion.

    :param parser: Argument parser used to report CLI usage errors.
    :type parser: argparse.ArgumentParser
    :param implicit_args: Tokens left unparsed after normal option parsing.
    :type implicit_args: list[str]
    :param no_typo_check: When true, accept implicit params even if they look
        similar to known workflow options.
    :type no_typo_check: bool
    :returns: Normalized ``key=value`` terms to append to ``args.param``.
    :rtype: list[str]
    :raises SystemExit: Raised indirectly via ``parser.error`` when leftover
        tokens are not valid implicit parameter flags.
    """
    terms = []
    known_long_options = _known_long_option_names(parser)
    index = 0
    while index < len(implicit_args):
        term = implicit_args[index]
        if term == "--":
            index += 1
            continue
        if not term.startswith("--"):
            parser.error("unrecognized arguments: {}".format(" ".join(implicit_args[index:])))
        stripped = term[2:]
        if not stripped:
            parser.error("unrecognized arguments: {}".format(term))
        option_name = stripped.split("=", 1)[0]
        if not no_typo_check:
            suggestion = get_close_matches(option_name, known_long_options, n=1, cutoff=0.75)
            if suggestion:
                display = term
                has_value = index + 1 < len(implicit_args)
                if "=" not in stripped and has_value and not implicit_args[index + 1].startswith("-"):
                    display = "{} {}".format(term, implicit_args[index + 1])
                parser.error(
                    "unrecognized arguments: {}. Did you mean --{}? Use --no-typo-check to force it through.".format(
                        display, suggestion[0]
                    )
                )
        if "=" in stripped:
            terms.append(stripped)
            index += 1
            continue
        if index + 1 >= len(implicit_args):
            parser.error("argument {}: expected one value".format(term))
        value = implicit_args[index + 1]
        if value.startswith("-"):
            parser.error("argument {}: expected one value".format(term))
        terms.append("{}={}".format(stripped, value))
        index += 2
    return terms


def main(argv=None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    actions = parser.add_argument_group("Actions")
    selection = parser.add_argument_group("Selection")
    enrichment = parser.add_argument_group("Enrichment")
    settings = parser.add_argument_group("Settings")

    actions.add_argument("--wipe", action="store_true", help="delete store contents before running")
    actions.add_argument("--delete-ensemble", action="store_true", help="delete ensemble if it exists")
    actions.add_argument(
        "--check",
        action="store_true",
        help=(
            "only check if matching datasets exist (exit 0 if yes, 1 if no). "
            "Prints one line per match: <id> <step> <step_last_modified_iso> <step_last_modified_epoch> "
            "(tab-separated)."
        ),
    )
    actions.add_argument(
        "--size",
        action="store_true",
        help="print number of datasets in the ensemble (or store if --ensemble omitted) and exit",
    )
    actions.add_argument("--step", help="workflow step value to set on matching datasets")
    actions.add_argument(
        "--init-step",
        default="init",
        help="step value that indicates a new dataset should be created",
    )
    upsert_group = actions.add_mutually_exclusive_group()
    upsert_group.add_argument(
        "--upsert",
        nargs="?",
        const=True,
        default=None,
        type=_str2bool,
        help="when step==init-step, do not create a duplicate dataset; update the existing match",
    )
    upsert_group.add_argument(
        "--upsert-init",
        nargs="?",
        const=True,
        default=None,
        type=_str2bool,
        help="alias for --upsert (deprecated)",
    )

    selection.add_argument(
        "--ensemble",
        "-e",
        default=None,
        help="ensemble name (omit to search/update across all datasets in the store)",
    )
    selection.add_argument(
        "--param",
        "-p",
        action="append",
        default=[],
        help=(
            "init parameter constraint as key=value (repeatable). "
            "Use this for parameter names with exotic characters; simple names can also be "
            "passed as --name=value."
        ),
    )
    selection.add_argument(
        "--strict-match",
        action="store_true",
        help="require exact numeric param matches (ignore --rtol/--atol; no DataRange matching)",
    )
    selection.add_argument(
        "--rtol",
        type=float,
        default=None,
        help="relative tolerance for numeric param matching (mutually exclusive with --strict-match)",
    )
    selection.add_argument(
        "--atol",
        type=float,
        default=None,
        help="absolute tolerance for numeric param matching (mutually exclusive with --strict-match)",
    )
    selection.add_argument(
        "--no-typo-check",
        action="store_true",
        help="accept implicit params even when they resemble known workflow options",
    )

    enrichment.add_argument("--meta", "-m", nargs=2, action="append", default=[],
                            help="dataset metadata update as key value (repeatable)")
    enrichment.add_argument(
        "--eme",
        "--emeta",
        "--ensemble-meta",
        dest="eme",
        nargs=2,
        action="append",
        default=[],
        help="ensemble metadata update as key value (repeatable)",
    )
    enrichment.add_argument("--associate", "-a", nargs=2, action="append", default=[],
                            help="associate file with dataset: path mime_type (repeatable)")
    enrichment.add_argument("--associate-ensemble", "--ae", nargs=2, action="append", default=[],
                            help="associate file with ensemble: path mime_type (repeatable)")

    settings.add_argument("--store", "-s", required=True, help="Kosh store to use")
    settings.add_argument(
        "--dataset_record_type",
        "-d",
        default=None,
        help=(
            "type used by sina db that Kosh will recognize as dataset; "
            "if omitted, searches span all non-reserved record types"
        ),
    )
    settings.add_argument("--step-field", default="workflow_step", help="field name to store the step value under")
    settings.add_argument(
        "--check-time-format",
        default="iso",
        help=(
            "format for the human-readable timestamp printed by --check. "
            "Use 'iso' for datetime.isoformat() (default) or pass a strftime() format string "
            "(e.g. '%%Y-%%m-%%d %%H:%%M:%%S')."
        ),
    )

    args, extra_args = parser.parse_known_args(raw_argv)
    args.param.extend(_collect_implicit_param_terms(parser, extra_args, no_typo_check=args.no_typo_check))
    dataset_record_type = args.dataset_record_type

    rtol_provided = args.rtol is not None
    atol_provided = args.atol is not None

    if args.rtol is None:
        args.rtol = 1e-5
    if args.atol is None:
        args.atol = 1e-8

    if args.strict_match and (rtol_provided or atol_provided):
        parser.error("--strict-match is mutually exclusive with --rtol/--atol")

    if not args.check and not args.size and not args.step:
        parser.error("--step is required unless using --check or --size")

    if args.ensemble is None:
        if args.delete_ensemble:
            parser.error("--delete-ensemble requires --ensemble")
        if args.associate_ensemble:
            parser.error("--associate-ensemble/--ae requires --ensemble")
        if args.eme:
            parser.error("--eme requires --ensemble")

    upsert = args.upsert
    if upsert is None:
        upsert = args.upsert_init
    if upsert is None:
        upsert = False

    if upsert and args.step != args.init_step:
        warnings.warn("--upsert is ignored unless --step matches --init-step value", RuntimeWarning)

    init_params = _parse_kv_pairs(args.param)
    dataset_associations = [Association(path=path, mime_type=mime) for path, mime in args.associate]
    ensemble_associations = [Association(path=path, mime_type=mime) for path, mime in args.associate_ensemble]

    if args.size:
        connect_kwargs = {}
        if dataset_record_type:
            connect_kwargs["dataset_record_type"] = dataset_record_type
        store = kosh.connect(args.store, **connect_kwargs)
        try:
            if args.ensemble is None:
                size = len(list(store.find(ids_only=True, **_dataset_type_filter_kwargs(dataset_record_type))))
                print(size)
                if size == 0:
                    raise SystemExit(1)
                return
            else:
                ensembles = list(store.find_ensembles(name=args.ensemble))
                if not ensembles:
                    print("0")
                    raise SystemExit(1)
                ensemble = ensembles[0]
                print(len(list(ensemble.find_datasets(ids_only=True))))
                return
        finally:
            store.close()

    if args.check:
        query = {}
        for key, value in init_params.items():
            if isinstance(value, (int, float)):
                value = float(value)
                if args.strict_match:
                    query[key] = value
                else:
                    lower = value - (args.atol + args.rtol * abs(value))
                    upper = value + (args.atol + args.rtol * abs(value))
                    if lower == upper:
                        query[key] = DataRange(lower, upper, max_inclusive=True)
                    else:
                        query[key] = DataRange(lower, upper)
            else:
                query[key] = value
        connect_kwargs = {}
        if dataset_record_type:
            connect_kwargs["dataset_record_type"] = dataset_record_type
        store = kosh.connect(args.store, **connect_kwargs)
        try:
            if args.ensemble is None:
                matches = list(store.find(ids_only=True, **_dataset_type_filter_kwargs(dataset_record_type), **query))
            else:
                ensembles = list(store.find_ensembles(name=args.ensemble))
                if not ensembles:
                    raise SystemExit(1)
                ensemble = ensembles[0]
                matches = list(ensemble.find_datasets(ids_only=True, **query))
            if matches:
                last_modified_key = f"{args.step_field}_last_modified"
                for match_id in matches:
                    record = store.get_record(match_id)
                    step_value = None
                    try:
                        step_value = record["data"][args.step_field]["value"]
                    except Exception:
                        step_value = None
                    try:
                        timestamp = record["user_defined"]["kosh_information"].get(last_modified_key)
                    except Exception:
                        timestamp = None
                    iso = ""
                    if isinstance(timestamp, (int, float)):
                        dt = datetime.fromtimestamp(float(timestamp))
                        if args.check_time_format == "iso":
                            iso = dt.isoformat()
                        else:
                            iso = dt.strftime(args.check_time_format)
                    print(f"{match_id}\t{step_value}\t{iso}\t{timestamp}")
                return
            raise SystemExit(1)
        finally:
            store.close()

    if upsert and args.step == args.init_step:
        # Provide a user-facing message about duplicate detection.
        # The actual upsert behavior is enforced in apply_step.
        query = {}
        for key, value in init_params.items():
            if isinstance(value, (int, float)):
                value = float(value)
                if args.strict_match:
                    query[key] = value
                else:
                    lower = value - (args.atol + args.rtol * abs(value))
                    upper = value + (args.atol + args.rtol * abs(value))
                    if lower == upper:
                        query[key] = DataRange(lower, upper, max_inclusive=True)
                    else:
                        query[key] = DataRange(lower, upper)
            else:
                query[key] = value
        connect_kwargs = {}
        if dataset_record_type:
            connect_kwargs["dataset_record_type"] = dataset_record_type
        store = kosh.connect(args.store, **connect_kwargs)
        try:
            if args.ensemble is None:
                matches = list(store.find(ids_only=True, **_dataset_type_filter_kwargs(dataset_record_type), **query))
                if matches:
                    print(f"Found existing dataset {matches[0]} matching init params; upserting it.")
            else:
                ensembles = list(store.find_ensembles(name=args.ensemble))
                if len(ensembles) == 1:
                    ensemble = ensembles[0]
                    matches = list(ensemble.find_datasets(ids_only=True, **query))
                    if matches:
                        print(f"Found existing dataset {matches[0]} matching init params; upserting it.")
        finally:
            store.close()

    request = StepRequest(
        store_uri=args.store,
        ensemble_name=args.ensemble,
        step=args.step,
        init_params=init_params,
        rtol=args.rtol,
        atol=args.atol,
        upsert_init=upsert,
        delete_ensemble=args.delete_ensemble,
        wipe_store=args.wipe,
        step_field=args.step_field,
        init_step_value=args.init_step,
        metadata_updates=dict(args.meta),
        dataset_associations=dataset_associations,
        ensemble_associations=ensemble_associations,
        ensemble_metadata_updates=dict(args.eme),
        strict_match=args.strict_match,
        dataset_record_type=dataset_record_type,
        connect_kwargs={"dataset_record_type": dataset_record_type} if dataset_record_type else {},
    )
    ids = apply_step(request)
    print("Matched dataset ids:", " ".join(ids))


if __name__ == "__main__":
    main()
