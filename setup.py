from setuptools import setup

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='gymplatformer',
    version='1.0',
    author='Aydens01',
    author_email='adrienlafage36@gmail.com',
    description='Platformer environment package',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=['gym', 'pygame']
)
