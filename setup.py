from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().split('\n')

setup(
    name='bit-algo-vis-tool',
    description='Laboratory classes in the Geometric Algorithms course for Computer Science students at the AGH University.',
    author='AGH BIT Student Scientific Group',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=requirements,
)
