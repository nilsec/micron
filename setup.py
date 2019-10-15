from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

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
        'micron.graph.ext',
        'micron.scripts',
        'micron.gp',
        'micron.solve',
        'micron.post',
            ],
     ext_modules=cythonize([
          Extension('micron.graph.ext.cpp_get_evidence',
          sources=[
              "micron/graph/ext/cpp_get_evidence.pyx",
              "micron/graph/ext/ext_cpp_get_evidence.cpp",
              ],
      extra_compile_args=['-O3', '-std=c++11'],
      include_dirs=[numpy.get_include()],
      language='c++')]), 
    install_requires = [
        'zarr',
        'daisy',
        'ConfigParser',
        'scipy',
        'pymongo',
        'ConfigArgParse',
        'click'
            ],
) 
