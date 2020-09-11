import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dialogpt2", # Replace with your own username
    version='1.0',
    author="ysig",
    author_email="yiannis@echochamber.be",
    description="DialoGTP2 train/gen",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dialogpt2",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=['dialogpt2'],
    license='Apache 2.0 Software license',
    scripts=['bin/dialogpt2-train', 'bin/dialogpt2-gen'],
    python_requires='>=3.5',
)
