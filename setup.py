from setuptools import setup, find_packages


setup(
    name='reconstruction_models',
    version='0.0.1',
    license='MIT',
    description='Reconstruction Models',
    packages=find_packages(),
    install_requires=[
        'torch',
    ],
)