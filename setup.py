#!/usr/bin/env python

from distutils.core import setup

setup(name='Equilid',
      version='1.0.1',
      description='Socially-Eqiutable Language Identification',
      author='David Jurgens and Yulia Tsvetkov',
      author_email='jurgens@stanford.edu',
      url='https://github.com/davidjurgens/equilid',
      packages=['equilid',],
      install_requires=[
        'tensorflow==0.11.0', 'numpy', 'regex'
      ],
      include_package_data=True,
      zip_safe=False,
      #data_files=[('70langs', ['models/70lang/checkpoint',
      #                         'models/70lang/equilid',
      #                         'models/70lang/equilid.meta',
      #                         'models/70lang/vocab.src',
      #                         'models/70lang/vocab.tgt', ])],
      keywords='langid language identification indigenous codeswitching code-switching',
      
      classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',

        'Topic :: Scientific/Engineering :: Human Machine Interfaces',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Text Processing :: Linguistic',
        
        'License :: OSI Approved :: Apache Software License',

        
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        ],
      
      entry_points= {
        'console_scripts': [
            'equilid = equilid.equilid:main',
            ],
        },
      )
