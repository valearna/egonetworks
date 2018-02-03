#!/usr/bin/env python3

from setuptools import setup

setup(name='egonetworks',
      version='0.1',
      description='Structural analysis of egocentric network graphs',
      url='',
      author='Valerio Arnaboldi',
      author_email='valerio.arnaboldi@iit.cnr.it',
      license='MIT',
      packages=['egonetworks'],
      setup_requires=['numpy', 'cython'],
      install_requires=[
          'jenks',
          'tweepy',
          'python-igraph',
          'sklearn',
          'scipy',
          'numpy'
      ],
      dependency_links=[
          'git+https://github.com/valearna/jenks.git#egg=jenks'
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)
