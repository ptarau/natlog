import subprocess
from sys import platform
import os
from setuptools import setup
import natlog

with open('requirements.txt') as f:
    required = f.read().splitlines()
with open("README.md","r") as f:
    long_description = f.read()

version = natlog.__version__
setup(name='Natlog',
  version=version,
  description='Prolog-like interpreter and tuple store',
  long_description = long_description,
  long_description_content_type='text/markdown',
  url='https://github.com/ptarau/minlog.git',
  author='Paul Tarau',
  author_USER_EMAIL='<paul.tarau@gmail.com>',
  license='Apache',
  packages=['Natlog'],
  package_data={'natprogs': ['*.nat',"*.json"], 'bak':["*.py"]},
  include_package_data=True,
  install_requires = required,
  zip_safe=False
  )
