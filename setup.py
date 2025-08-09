#!/usr/bin/env python3
"""
Setup script for the RFM package.
"""

from setuptools import setup, find_packages

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="rfm",
    version="0.0.1",
    author="Anthony Liang",
    author_email="aliang80@usc.edu",
    description="PyTorch implementation of a Reward Foundation Model (RFM)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aliang8/reward_fm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "rfm-visualizer=rfm_visualizer.app:main",
            "rfm-train=train:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
) 