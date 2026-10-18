from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="ancestryaudit",
    version="0.4.0",
    author="Dana Yergaliyeva",
    description="Explains accuracy gaps between populations before correcting genomic cancer models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DanYerga/ancestryaudit",
    project_urls={
        "Source": "https://github.com/DanYerga/ancestryaudit",
        "Validation notebook": "https://www.kaggle.com/code/danae345/project-final",
        "Bug tracker": "https://github.com/DanYerga/ancestryaudit/issues",
    },
    license="MIT",
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
