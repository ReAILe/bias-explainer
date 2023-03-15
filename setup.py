from setuptools import setup

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# with open('requirements.txt') as f:
#     required = f.read().splitlines()


setup(
  name = 'fairxplainer',
  packages = ['fairxplainer', 'fairxplainer.salib_util', 'fairxplainer.wrapper'],
  version = 'v0.2.0',
  license='MIT',
  description = 'A Python package for explaining bias in machine learning models',
  long_description=long_description,
  long_description_content_type='text/markdown',
  author = 'Bishwamittra Ghosh',
  author_email = 'bishwamittra.ghosh@gmail.com',
  url = 'https://github.com/ReAILe/bias-explainer',
  download_url = 'https://github.com/ReAILe/bias-explainer/archive/v0.2.0.tar.gz',
  keywords = ['Fair Machine Learning', 'Bias', 'Explainability', 'Global Sensitivity Analysis', 'Variance Decomposition', 'Influence Functions'],   # Keywords that define your package best
  # install_requires=required,
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7',
  ],
)