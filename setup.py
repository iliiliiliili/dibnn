# python3
# pylint: disable=g-bad-file-header
# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Install script for setuptools."""

import imp

import setuptools

# Additional requirements for testing.
testing_require = [
    "mock",
    "pytest-xdist",
    "pytype",
]

setuptools.setup(
    name="dm-enn",
    description=(
        "Epistemic neural networks. "
        "A library for probabilistic inference via neural networks."
    ),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/deepmind/enn",
    author="DeepMind",
    author_email="enn-eng+os@google.com",
    license="Apache License, Version 2.0",
    version=imp.load_source("_metadata", "enn/_metadata.py").__version__,
    keywords="probabilistic-inference python machine-learning",
    packages=setuptools.find_packages(),
    install_requires=[
        "absl-py",
        "chex==0.0.8",
        "dm-acme==0.2.2",
        "dm-haiku==0.0.5.dev0",
        "dataclasses",  # Back-port for Python 3.6.
        "jax==0.2.20",
        "jaxlib @ https://storage.googleapis.com/jax-releases/nocuda/jaxlib-0.1.71-cp38-none-manylinux2010_x86_64.whl",
        "matplotlib==3.4.3",
        "neural-tangents==0.3.7",
        "numpy==1.19.5",
        "optax==0.0.9",
        "pandas==1.3.3",
        "rlax==0.0.4",
        "plotnine==0.8.0",
        "scipy==1.7.1",
        "scikit-image",
        "scikit-learn",
        "six",
        "tensorflow-gpu==2.6.0",
        "tensorflow-datasets==4.4.0",
        "termcolor",
        "typing-extensions",
        "tf-keras",
        "opencv-python",
        "plotnine"
    ],
    extras_require={"testing": testing_require,},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
