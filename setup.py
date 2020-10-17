import pathlib
from setuptools import setup, find_packages

base_packages = [
    "gensim>=3.8.3",
    "torch>=1.6.0",
    "sentencepiece>=0.1.91",
    "tokenizers>=0.9.2"
]

docs_packages = [
    "mkdocs==1.1",
    "mkdocs-material==4.6.3",
    "mkdocstrings==0.8.0",
    "jupyterlab>=0.35.4",
    "nbstripout>=0.3.7",
    "nbval>=0.9.5",
]

test_packages = [
    "flake8>=3.6.0",
    "pytest>=4.0.2",
    "black>=19.3b0",
    "pytest-cov>=2.6.1",
    "nbval>=0.9.5",
    "pre-commit>=2.2.0",
]

dev_packages = docs_packages + test_packages


setup(
    name="inlay",
    version="0.1.0",
    author="Vincent D. Warmerdam",
    packages=find_packages(exclude=["notebooks", "docs"]),
    description="Ideas for custom embeddings.",
    long_description=pathlib.Path("readme.md").read_text(),
    long_description_content_type="text/markdown",
    install_requires=base_packages,
    extras_require={
        "base": base_packages,
        "docs": docs_packages,
        "dev": dev_packages,
        "test": test_packages,
    },
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
