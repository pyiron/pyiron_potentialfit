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
        'ase==3.23.0',
        'matplotlib>=3.8.4,<4',
        'numpy==1.26.4',
        'pyiron_base==0.9.12',
        'pyiron_atomistics==0.6.10',
        'scipy==1.14.0',
        'runnerase==0.3.3',
        'pyiron_snippets>=0.1.1,<=0.1.4',
        'pyparsing>=3.1.2,<3.2',
        # spgfit
        'pyiron_contrib==0.1.18',
        'dill>=0.3.0',
        'seaborn>=0.13.0,<0.14',
        'pyxtal>=0.6.0,<0.7',
    ],
    cmdclass=versioneer.get_cmdclass(),
    entry_points={
        "console_scripts": [
            "spgfit-structures = pyiron_potentialfit.spgfit.structures:main",
            "spgfit-calculations = pyiron_potentialfit.spgfit.calculations:main",
            "spgfit-learn = pyiron_potentialfit.spgfit.learn:main"
        ]
    }
)
