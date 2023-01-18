from setuptools import setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

exec(open("pycircstat2/version.py").read())
setup(
    name="pycircstat2",
    version=__version__,  # noqa: F821
    description="Python toolbox for circular statistcs",
    author=["Ziwei Huang", "Philipp Berens"],
    author_email="huang-ziwei@outlook.com",
    install_requires=required,
    packages=["pycircstat2"],
    include_package_data=True,
    package_data={"pycircstat2": ["data/fisher_1993/*.csv", "data/fisher_1993/*.json"]},
)
