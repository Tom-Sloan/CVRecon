from setuptools import setup, find_packages

setup(
    name="cvrecon",
    version="0.1.0",
    author="Tom Sloan",
    author_email="tom.sloan",
    description="A 3D reconstruction module using cost volume techniques",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "torch",
        "torchvision",
        "torchsparse",
        "pytorch-lightning",
        "open3d",
        "scikit-image",
        "einops",
        "antialiased_cnns",
    ],
    extras_require={
        "dev": [
            "pytest",
            "flake8",
            "black",
        ],
    },
)
