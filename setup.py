from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip()
                    and not line.startswith("#")]

setup(
    name="ancestryaudit",
    version="0.1.0",
    author="Dana Yergaliyeva",
    author_email="",
    description=(
        "Bias detection and correction framework for genomic cancer AI. "
        "Detects ancestry-linked performance gaps in CNV-based cancer "
        "classifiers and applies supervised fine-tuning correction."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/danayergaliyeva/ancestryaudit",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    keywords=[
        "bioinformatics", "cancer genomics", "copy number variation",
        "ancestry bias", "fairness", "machine learning", "TCGA"
    ],
)
