from setuptools import setup
import setuptools

import natlog

#with open('natlog/requirements.txt') as f:
#    required = f.read().splitlines()
required = []
with open("README.md", "r") as f:
    long_description = f.read()

version = natlog.__version__
setup(name='natlog',
      version=version,
      description='Prolog-like interpreter and tuple store',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/ptarau/natlog.git',
      author='Paul Tarau',
      author_email='paul.tarau@gmail.com',
      license='Apache',
      packages=setuptools.find_packages(),
      package_data={'natlog': [
                               'natlog/requirements.txt',
                               'natprogs/*.nat',
                               'natprogs/*.tsv',
                               'natprogs/*.pro',
                               'natprogs/*.json'
                               ]
                    },
      include_package_data=True,
      install_requires=required,
      zip_safe=False
      )
