import setuptools
import os

PACKAGE = "TFGENZOO"

about = {}
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, PACKAGE, "__version__.py")) as f:
    exec(f.read(), about)

setuptools.setup(
    name=PACKAGE,
    packages=setuptools.find_packages(),
    install_requires=["tensorflow>=2.1.0", "tensorflow_probability>=0.9.0", "pandas"],
    version=about["__version__"],
    author="MokkeMeguru",
    author_email="meguru.mokke@gmail.com",
    description="helper of building generative model with Tensorflow 2.x",
    long_description=(
        "heavily documented (test and description and formula)"
        + "generative model examples / layers"
    ),
    long_description_content_type="text/markdown",
    url="https://github.com/MokkeMeguru/TFGENZOO",
    licence="MIT",
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Environment :: GPU :: NVIDIA CUDA :: 10.0",
        "Environment :: GPU :: NVIDIA CUDA :: 10.1",
    ],
)
