#!/usr/bin/env python
"""Apply or verify the upstream MatterSim PR #166 batching fix.

Affected released versions: 1.2.4 and 1.2.5.
The patch changes GraphConverter.convert() to return M3GNetData so PyG offsets
three_body_indices by the previous graph's num_bonds during batching.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import shutil
from datetime import datetime
from pathlib import Path


BUGGY_LINE = "            return Data(**args)"
FIXED_LINE = "            return M3GNetData(**args)"
AFFECTED_VERSIONS = {"1.2.4", "1.2.5"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def converter_path() -> Path:
    import mattersim.datasets.utils.converter as converter

    return Path(converter.__file__).resolve()


def regression_test() -> None:
    import torch
    from ase.build import bulk
    from mattersim.datasets.utils.converter import GraphConverter, M3GNetData
    from torch_geometric.data import Batch

    converter = GraphConverter(model_type="m3gnet")
    structure = bulk("Si", "diamond", a=5.43)
    graph0 = converter.convert(structure)
    graph1 = converter.convert(structure.copy())
    if not isinstance(graph0, M3GNetData):
        raise RuntimeError(f"wrong_graph_type:{type(graph0).__name__}")
    batch = Batch.from_data_list([graph0, graph1])
    first_count = int(graph0.num_three_body)
    second_count = int(graph1.num_three_body)
    actual = batch.three_body_indices[first_count : first_count + second_count]
    expected = graph1.three_body_indices + graph0.num_bonds
    if not torch.equal(actual, expected):
        raise RuntimeError(
            f"three_body_index_offset_failed:actual={actual[0].tolist()}:"
            f"expected={expected[0].tolist()}"
        )
    print(f"graph_type={type(graph0).__name__}")
    print(f"num_bonds_graph0={int(graph0.num_bonds)}")
    print(f"actual_first={actual[0].tolist()}")
    print(f"expected_first={expected[0].tolist()}")
    print("PR166_REGRESSION_TEST=PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--force-unknown-version", action="store_true")
    args = parser.parse_args()

    version = importlib.metadata.version("mattersim")
    path = converter_path()
    text = path.read_text(encoding="utf-8")
    print(f"mattersim_version={version}")
    print(f"converter_path={path}")
    print(f"sha256_before={sha256(path)}")

    fixed = FIXED_LINE in text and BUGGY_LINE not in text
    if args.check_only:
        print(f"patch_present={fixed}")
        regression_test()
        return

    if version not in AFFECTED_VERSIONS and not args.force_unknown_version:
        raise SystemExit(
            f"Refusing to patch unrecognized MatterSim version {version}; "
            "use --force-unknown-version only after reviewing upstream source."
        )
    if fixed:
        print("patch_status=already_applied")
        regression_test()
        return
    occurrences = text.count(BUGGY_LINE)
    if occurrences != 1:
        raise RuntimeError(f"expected_one_buggy_line:found={occurrences}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(path.name + f".pre_pr166_{timestamp}.bak")
    shutil.copy2(path, backup)
    path.write_text(text.replace(BUGGY_LINE, FIXED_LINE, 1), encoding="utf-8")
    print(f"backup={backup}")
    print(f"sha256_after={sha256(path)}")
    regression_test()


if __name__ == "__main__":
    main()
