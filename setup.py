import os
from setuptools import setup, find_packages


def read_requirements():
    reqs_path = os.path.join('.', 'requirements.txt')
    with open(reqs_path, 'r') as f:
        requirements = [line.rstrip() for line in f]
    return requirements


setup(
    name="sapphire",
    version="0.1.0",
    description="Simple Aligner for Phrasal Paraphrase with Hierarchical Representation",
    author="Masato Yoshinaka",
    author_email="yoshinaka.masato@ist.osaka-u.ac.jp",
    install_requires=read_requirements(),
    url="https://github.com/mybon13/sapphire",
    packages=find_packages()
)