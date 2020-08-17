#!/usr/bin/env python3

import io
import os
import re

from setuptools import find_packages, setup


# Get version
def read(*names, **kwargs):
    with io.open(os.path.join(os.path.dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


# readme = open("README.md").read()  # .orgがpypiでは対応していない
version = find_version("eagpytorch", "__init__.py")

torch_min = "1.5"
install_requires = [
    "gpytorch",
    "matplotlib",
    "numpy",
    "pyro-ppl",
    "PyYAML"
    "attrdict"
]
try:
    import torch

    if torch.__version__ >= torch_min:
        install_requires = []
except ImportError:
    pass


# Run the setup
setup(
    name="eagpytorch",
    version=version,
    description="ガウス過程モデルを実行するためのモジュールを作成しました。 GPyTorchをベースにコードを実装しています。",
    # long_description=readme,  # .orgがpypiでは対応していない
    # long_description_content_type="text/markdown",  # .orgがpypiでは対応していない
    author="Hirotaka Kato, Naofumi Emoto and Yuya Kaneta",
    url="https://github.com/yucho147/GP",
    author_email="yucho147@gmail.com",
    project_urls={
        # "Documentation": "",
        "Source": "https://github.com/yucho147/GP",
    },
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=install_requires,
    extras_require={
        "dev": ["twine", "wheel"],
        "docs": ["sphinx", "sphinx_rtd_theme"],
    },
    keywords=["gpytorch", "pytorch", "gaussian-process"],
)
