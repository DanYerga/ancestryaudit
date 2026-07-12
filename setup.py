from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="ancestryaudit",
    version="0.3.9",
    author="Dana Yergaliyeva",
    description="Bias detection and correction framework for genomic cancer AI",
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
    ],
    python_requires=">=3.8",
    install_requires=[
        "scikit-learn>=1.0",
        "numpy>=1.21",
        "pandas>=1.3",
        "scipy>=1.7",
        "shap>=0.41",
        "matplotlib>=3.4",
    ],
)
