from os.path import join, dirname, realpath
from setuptools import setup
import sys

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, \
    "You should have Python 3.6 and greater." 

setup(
    name='thrifty',
    py_modules=['thrifty'],
    version='0.0.1',
    install_requires=[
        'mujoco-py==2.0.2.9',
        'cloudpickle',
        'gym',
        'joblib',
        'matplotlib',
        'numpy',
        'pandas',
        'pytest',
        'psutil',
        'scipy',
        'seaborn==0.8.1',
        'torch==1.6.0',
        'tqdm',
        'moviepy',
        'opencv-python',
        'torchvision==0.7.0',
        'robosuite==1.2.0',
        'h5py',
        'hidapi',
        'pygame'
    ],
    description="Code for ThriftyDAgger.",
    author="Ryan Hoque",
)