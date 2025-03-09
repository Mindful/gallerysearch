from setuptools import setup, find_packages

def read_requirements():
    with open('requirements.txt') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

setup(
    name='gallerysearch',
    version='0.1',
    packages=find_packages(),
    install_requires=read_requirements(),
    entry_points={
        'console_scripts': [
            'galsearch=gallerysearch.cli:main',  # Replace with your actual entry point
        ],
    },
)
