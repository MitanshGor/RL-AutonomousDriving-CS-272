"""
Setup script for RL-AutonomousDriving-CS-272
This script is used to setup the project environment.

Usage:
    python setup.py install

Requirements:
    - setuptools
    - pathlib
    - readme_file
    - requirements_file
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip()
        for line in requirements_file.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="rl-autonomous-driving",
    version="1.0.0",
    description="Reinforcement Learning project for autonomous driving using HighwayEnv",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Mitansh Gor, Henry Ha, John Yun",
    packages=find_packages(exclude=["tests", "examples", "docs"]),
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
)

