"""
Setuptools based setup module
"""
from setuptools import setup, find_packages
import versioneer


setup(
    name='pyiron_potentialfit',
    version=versioneer.get_version(),
    description='Repository for user-generated plugins to the pyiron IDE.',
    long_description='http://pyiron.org',

    url='https://github.com/pyiron/pyiron_potentialfit',
    author='Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department',
    author_email='huber@mpie.de',
    license='BSD',

    classifiers=[
        'Development Status :: 4 - Beta',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: BSD License',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],

    keywords='pyiron',
    packages=find_packages(exclude=["*tests*"]),
    install_requires=[
        'ase==3.22.1',
        'pyiron_atomistics==0.3.7',
        'matplotlib==3.8.2',
        'numpy==1.26.2',
        'pyiron_base==0.6.16',
        'scipy==1.11.4',
    ],
    cmdclass=versioneer.get_cmdclass(),
    
)
