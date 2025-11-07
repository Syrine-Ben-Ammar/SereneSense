#!/usr/bin/env python
"""Setup configuration for SereneSense package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="core",
    version="1.0.0",
    author="Syrine Ben Ammar",
    author_email="sirine.ben.ammar32@gmail.com",
    description="Enterprise-grade AI system for real-time military vehicle sound detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Syrine-Ben-Ammar/SereneSense",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "core-train=core.scripts.train_model:main",
            "core-eval=core.scripts.evaluate_model:main",
            "core-optimize=core.scripts.optimize_for_edge:main",
            "core-deploy=core.scripts.deploy_model:main",
            "core-serve=core.deployment.api.fastapi_server:main",
            "core-realtime=core.inference.realtime.detector:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
