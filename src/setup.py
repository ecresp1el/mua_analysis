from setuptools import setup, find_packages
import subprocess
import os

# Run the setup_kernel.py script to set up the Jupyter kernel
kernel_script_path = os.path.join(os.path.dirname(__file__), "../scripts/setup_kernel.py")

subprocess.run(["python", kernel_script_path])

setup(
    name="neural_analysis_pkg",
    version="0.1.1",  
    author="Emmanuel Crespo",
    author_email="cresp1el@cmich.edu",
    description="A package for neural analysis based on extracellular recordings",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/ecresp1el/mua_analysis",  # Your repository URL
    packages=find_packages(),  # Removed where argument
    # Removed package_dir line
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # or another appropriate license
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy>=1.24.3",
        "scipy>=1.10.1",
        "ipykernel",
    ],
    python_requires='>=3.8.18',
)


