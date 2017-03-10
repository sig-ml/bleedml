from setuptools import setup
from bleedml import __version__
__version__ = list(map(str, __version__))
setup(name='bleedml',
        version='.'.join(__version__),
        description='Bleeding edge Machine Learning Algorithms',
        url='http://github.com/sig-ml/bleedml',
        author='Arjoonn Sharma',
        author_email='arjoonn.94@gmail.com',
        license='MIT',
        packages=['bleedml'],
        install_requires=['numpy', 'scipy', 'sklearn'],
        zip_safe=False)
