from setuptools import setup
import setuptools

import natlog

with open('requirements.txt') as f:
    required = f.read().splitlines()
with open("README.md", "r") as f:
    long_description = f.read()

version = natlog.__version__
setup(name='natlog',
      version=version,
      description='Prolog-like interpreter and tuple store',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/ptarau/minlog.git',
      author='Paul Tarau',
      author_email='paul.tarau@gmail.com',
      license='Apache',
      packages=setuptools.find_packages(),
      package_data={'natlog': [
                               'natprogs/*.nat',
                               'natprogs/*.tsv',
                               'natprogs/*.pro',
                               'natprogs/*.json',
                               'doc/*.pdf',
                               ]
                    },
      include_package_data=True,
      # scripts=['bin/*'],
      install_requires=required,
      zip_safe=False
      )
