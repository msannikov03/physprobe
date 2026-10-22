"""Repository paths. numpy-free, stdlib only. See CONTRACT.md."""
import os


def code_root():
    """The clone that holds this package (used for git provenance)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def root():
    """$PHYSPROBE_ROOT if set, else the clone root (CONTRACT v1)."""
    r = os.environ.get("PHYSPROBE_ROOT")
    if r:
        return os.path.abspath(r)
    return code_root()


def data_dir():
    return os.path.join(root(), "data")


def features_dir():
    return os.path.join(root(), "features")


def results_dir(name=None):
    d = os.path.join(root(), "results")
    return os.path.join(d, name) if name else d


def dataset_dir(name):
    return os.path.join(data_dir(), name)


def features_path(name, rep, mode):
    return os.path.join(features_dir(), name, rep, f"{mode}.npz")
