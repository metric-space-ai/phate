from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="phate",
    version="1.0.11",
    author="Michael Welsch",
    author_email="michael.welsch@metric-space.ai",
    description="Implementation of the PHATE algorithm",
    long_description='Refactored version of the PHATE algorithm, originally implemented by Krishnaswamy Labs, that does no use outdated dependencies any more'
    long_description_content_type="text/markdown",
    url="https://github.com/metric-space-ai/phate",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "graphtools",
    ],
)
