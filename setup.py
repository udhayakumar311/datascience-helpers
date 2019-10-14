from setuptools import setup

version = '0.1.0'

setup(
    name='datascience-helpers',
    packages=['datascience-helpers'], # this must be the same as the name above
    install_requires=['numpy', 'matplotlib', 'scipy', 'seaborn', 'missingno'],
    extras_require={'tests': ['pytest', 'pytest-mpl']},
    py_modules=['datascience-helpers'],
    version=f'{version}',  # note to self: also update the one is the source!
    description='A library with helpful functions and classes that help me do my job with less copy-pasting.',
    author='Mycchaka Kleinbort',
    author_email='mkleinbort@gmail.com',
    url='https://github.com/mkleinbort/datascience-helpers.git',
    download_url='https://github.com/mkleinbort/datascience-helpers/tarball/{version}',
    keywords=['data', 
              'data visualization', 
              'data analysis', 
              'data science',
              'pandas', 
              'python',
              'jupyter'],
    license='GNU',
    classifiers=[]
)