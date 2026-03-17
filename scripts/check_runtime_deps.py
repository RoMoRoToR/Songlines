#!/usr/bin/env python3
import importlib
import sys


REQUIRED = [
    ("numpy", "numpy"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("matplotlib", "matplotlib"),
    ("gym", "gym"),
    ("cv2", "opencv-python"),
    ("quaternion", "numpy-quaternion"),
    ("skimage", "scikit-image"),
    ("sklearn", "scikit-learn"),
    ("skfmm", "scikit-fmm"),
    ("yacs", "yacs"),
    ("numba", "numba"),
    ("habitat", "habitat-api (editable install)"),
    ("habitat_sim", "habitat-sim"),
]


def main():
    missing = []
    print("Python:", sys.version.replace("\n", " "))
    print("Executable:", sys.executable)
    print()
    for mod_name, pkg_name in REQUIRED:
        try:
            importlib.import_module(mod_name)
            print("[OK]      {:<16} ({})".format(mod_name, pkg_name))
        except Exception as exc:
            print("[MISSING] {:<16} ({}) -> {}".format(
                mod_name, pkg_name, exc.__class__.__name__
            ))
            missing.append((mod_name, pkg_name, str(exc)))

    print()
    if missing:
        print("Missing {} modules. Install these before baseline run:".format(
            len(missing)
        ))
        for _, pkg_name, _ in missing:
            print(" - {}".format(pkg_name))
        sys.exit(1)

    print("All runtime dependencies are importable.")


if __name__ == "__main__":
    main()
