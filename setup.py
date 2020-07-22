from setuptools import find_packages, setup

setup(
    name="deep_control",
    version="0.0.1",
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    description="Deep Reinforcement Learning for Continuous Control Tasks",
    author="Jake Grigsby",
    author_email="jcg6dn@virginia.edu",
    license="MIT",
    packages=find_packages(),
)
