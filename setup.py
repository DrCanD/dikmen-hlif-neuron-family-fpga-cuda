"""
H-LIF Neuron Family — FPGA & CUDA Reference Implementation
Patent: TÜRKPATENT 2026/007632
"""
from setuptools import setup, find_packages
from pathlib import Path

readme = Path(__file__).parent / "README.md"
long_desc = readme.read_text(encoding="utf-8") if readme.exists() else ""

setup(
    name="hlif-neuron-family",
    version="1.0.0",
    author="İsmail Can Dikmen",
    author_email="ismail.dikmen@istinye.edu.tr",
    description="Patent-scoped reference implementation of the H-LIF spiking neuron family (TÜRKPATENT 2026/007632), with custom CUDA kernel and FPGA measurements.",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    url="https://github.com/DrCanD/dikmen-hlif-neuron-family-fpga-cuda",
    license="Apache-2.0",
    packages=find_packages(exclude=["tests", "examples", "docs"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10",
        "numpy>=1.20",
    ],
    extras_require={
        "cuda": ["cupy-cuda12x"],  # for the custom TH-LIF CUDA kernel
        "dev": ["pytest>=7.0", "pytest-cov"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    project_urls={
        "Patent (TÜRKPATENT 2026/007632)": "https://www.turkpatent.gov.tr/",
        "Parent library": "https://github.com/DrCanD/dikmen-spiking-neurons",
        "Author profile": "https://orcid.org/0000-0002-7747-7777",
    },
)
