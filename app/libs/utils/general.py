"""General utility functions"""
import glob
import re
from pathlib import Path
from typing import Union


def increment_path(
    path: Union[str, bytes, Path],
    exist_ok: bool = False,
    sep: str = '',
    mkdir: bool = False
) -> Path:
    """
    Increment file or directory path.

    i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    """
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    directory = path if path.suffix == '' else path.parent  # directory
    if not directory.exists() and mkdir:
        directory.mkdir(parents=True, exist_ok=True)  # make directory
    return path
