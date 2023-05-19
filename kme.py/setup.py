from setuptools import setup

meta = {}
with open('./__version__.py', encoding='utf-8') as f:
    exec(f.read(), meta)


setup(
  name = 'torchkme',
  packages = ['torchkme'],
  version=meta['__version__'],
  author="Andrea Pinto",
  author_email="andreakiro.pinto@gmail.com",
  python_requires=">=3.9"
)
