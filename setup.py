from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="legislation_analysis",
    version="0.1",
    packages=find_packages(),
    description="A computational content analysis/NLP package for analyzing abortion-related legislation and SCOTUS opinions.",
    long_description=open("README.md").read(),
    author="Chanteria Milner, Michael Plunkett",
    author_email="chanteria.milner@gmail.com, michplunkett@gmail.com",
    url="https://github.com/michplunkett/abortion-legislation-analysis",
    install_requires=requirements,
    classifiers=[
        "License :: OSI Approved :: MIT License",
    ],
)
