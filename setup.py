from setuptools import setup

def requirements():
    with open('requirements.txt', 'r') as reqs:
        return [line.strip() for line in reqs if not line.startswith('#')]

setup(
  name = 'rum',
  packages=['rum'],
  package_data={},
  install_requires=requirements()
)
