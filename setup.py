from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="ancestryaudit",
    version="0.1.0",
    author="Dana Yergaliyeva",
    author_email="dyergaliyeva.08@gmail.com",
    description=(
        "Bias detection and correction framework for genomic cancer AI. "
        "Detects ancestry-linked performance gaps in CNV-based cancer "
        "classifiers and applies supervised fine-tuning correction."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DanYerga/ancestryaudit",
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
    install_requires=[
        "numpy>=1.21",
        "pandas>=1.3",
        "scikit-learn>=1.0",
        "scipy>=1.7",
        "shap>=0.41",
        "matplotlib>=3.4",
    ],
    keywords=[
        "bioinformatics", "cancer genomics", "copy number variation",
        "ancestry bias", "fairness", "machine learning", "TCGA"
    ],
)
