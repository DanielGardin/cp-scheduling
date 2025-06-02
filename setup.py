import tomllib  # Use tomli if using Python < 3.11
from setuptools import setup
from mypyc.build import mypycify

# Load metadata from pyproject.toml
with open('pyproject.toml', 'rb') as f:
    project_info = tomllib.load(f)['project']

setup(
    name=project_info['name'],
    version=project_info['version'],
    description=project_info['description'],
    author=project_info['authors'][0]['name'],
    author_email=project_info['authors'][0]['email'],
    packages=['cpscheduler'], 
    install_requires=project_info.get('dependencies', []),
    tests_require=['pytest'],
    include_package_data=True,
    ext_modules=mypycify([
        "cpscheduler/environment/variables.py",
        "cpscheduler/environment/objectives.py",
        "cpscheduler/environment/constraints.py",
        "cpscheduler/environment/env.py",
    ]),
    classifiers=[
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
