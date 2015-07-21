from __future__ import with_statement
import os
from setuptools import find_packages
from distutils.core import setup


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "ENGAGE HACKER REPO",
    version = "0.1.0",
    author = "ENGAGE TEAM",
    author_email = "k.jones@ucl.ac.uk",
    description = ("Azzurroooo!"),
    packages=find_packages(),
    #license = read('LICENSE.txt'),
    keywords = "audio",
    url = "https://github.com/groakat/engaged_hackathon",
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
    ]
)
