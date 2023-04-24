from setuptools import setup, find_packages
import os
import json

with open("requirements.txt") as f:
    requirements = f.read().splitlines()


def get_long_description():
    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md"),
        encoding="utf8",
    ) as fp:
        return fp.read()


with open("package_setup.json", "r") as f:
    package_setup = json.load(f)
    package_name = package_setup["package_name"]
    package_version = package_setup["package_version"]
    package_description = package_setup["package_description"]
    package_tasks = package_setup["tasks"]


setup(
    name="hybrid_recommender",
    description="A package for classification ",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="Harish Nedunuri",
    url="hybrid_recommender-devops url",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            f"{package_name}.{task}={package_name}.{task}.entry:main"
            for task in package_tasks
        ]
    },
    version=package_version,
    install_requires=requirements,
    python_requires=">=3.8",
)
