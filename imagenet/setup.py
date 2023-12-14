from setuptools import setup, find_packages

import torchdistill

with open('README.md', 'r') as f:
    long_description = f.read()

description = 'A Modular, Configuration-Driven Framework for Knowledge Distillation. ' \
              'Trained models, training logs and configurations are available for ensuring the reproducibility.'
setup(
    name='torchdistill',
    version=torchdistill.__version__,
    author='Yoshitomo Matsubara',
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yoshitomo-matsubara/torchdistill',
    packages=find_packages(exclude=('tests', 'examples', 'demo', 'configs')),
    python_requires='>=3.7',
    install_requires=[
        'torch>=1.11.0',
        'torchvision>=0.12.0',
        'numpy',
        'pyyaml>=5.4.1',
        'scipy',
        'cython',
        'pycocotools>=2.0.2'
    ],
    extras_require={
        'test': ['pytest']
    }
)
