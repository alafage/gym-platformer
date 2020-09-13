from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="gymplatformer",
    version="1.0",
    author="Adrien Lafage",
    author_email="adrienlafage@outlook.com",
    description="Platformer environment package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=["gym", "pygame"],
    packages=find_packages(),
)
