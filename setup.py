from setuptools import setup, find_packages
import os

# Function to read dependencies from requirements.txt
def load_requirements(filename='requirements.txt'):
    try:
        with open(os.path.join(os.path.dirname(__file__), filename), 'r') as f:
            # Filter out comments and empty lines
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except IOError:
        return []

setup(
    name="hta_evaluation_harness",
    version="0.1.0",
    description="A harness for evaluating Hierarchical Task Analysis (HTA) based on 5 metrics.",
    author="Your Name / Team Name",
    packages=find_packages(include=['hta_evaluation_harness', 'hta_evaluation_harness.*']),
    install_requires=load_requirements('requirements.txt'),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License", # Example license
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
