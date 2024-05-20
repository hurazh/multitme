from setuptools import setup, find_packages

setup(
    name='my_package',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'Pillow',
        'matplotlib',
        'numpy',
        'pandas',
        'pyro-ppl',
        'readimc',
        'scipy',
        'torch',
        'tqdm',
    ],
    author='Haoran Zhang',
    author_email='hz6453@utexas.edu',
    description='multiTME',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/hurazh/multitme',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)