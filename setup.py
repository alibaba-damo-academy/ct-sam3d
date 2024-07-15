from setuptools import find_packages, setup

import sys


def parse_requirements(filename="requirements.txt"):
    # Read requirements.txt, ignore comments
    try:
        requirement_list = list()
        with open(filename, "rb") as f:
            lines = f.read().decode("utf-8").split("\n")
        for line in lines:
            line = line.strip()
            if "#" in line:
                # remove package starting with '#'
                line = line[: line.find("#")].strip()
            if line:
                if line.startswith("opencv-python"):
                    # in case of conda installed opencv, skip installing with pip
                    try:
                        import cv2

                        print(cv2.__version__)
                        continue
                    except Exception:
                        pass

                requirement_list.append(line)

    except Exception:
        print(f"'{filename}' not found!")
        requirement_list = list()
    return requirement_list


required_packages = parse_requirements()

setup(
    name="ct_sam",
    version="0.1",
    description="segment anything model on CT scan",
    author="MIA group",
    packages=find_packages(),
    install_requires=required_packages,
    dependency_links=[],
    include_package_data=True,
    zip_safe=False,
    test_suite="nose.collector",
    tests_require=["nose"],
    entry_points={
        "console_scripts": [
            "ct_sam_train=ct_sam.utils.cli_functions:ct_sam_train_dist",
            "ct_sam_train_cpp=ct_sam.utils.cli_functions:ct_sam_train_cpp_dist",
            "ct_sam_test=ct_sam.test:main",
            "ct_sam_valid=ct_sam.valid:main",
        ]
    },
)
