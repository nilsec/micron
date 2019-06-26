from setuptools import setup

setup(
    name='micron',
    version='0.1',
    description='ILP based tracking of microtubules in EM image stacks',
    url='https://github.com/nilsec/micron',
    author='Nils Eckstein',
    author_email='ecksteinn@janelia.hhmi.org',
    license='MIT',
    packages=[
        'micron',
        'micron.network',
        'micron.graph',
        'micron.scripts',
        'micron.gunpowder',
        'micron.solve'
            ],
    install_requires = [
        'zarr',
        'daisy',
        'ConfigParser',
        'neuroglancer',
        'funlib.show.neuroglancer',
        'scipy',
        'pymongo',
        'ConfigArgParse',
        'click'
            ],
) 
