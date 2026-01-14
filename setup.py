#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PU-GRAIL: Setup script for package installation
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pugrail",
    version="1.0.0",
    author="Jaemin Jeon",
    author_email="inukjung@snu.ac.kr",
    description="Graph Neural Network for Protective Antigen Prediction under PU Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jaeminjj/PU-GRAIL",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "torch-geometric>=2.3.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "biopython>=1.79",
    ],
    extras_require={
        "esm": ["fair-esm>=2.0.0"],
    },
    entry_points={
        "console_scripts": [
            "pugrail-train=pugrail.train:main",
            "pugrail-predict=pugrail.predict:main",
        ],
    },
)

