from setuptools import setup, find_packages

setup(
    name="aznas",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "scipy"
    ],
    author="Tan Xinyuan",
    description="Zero-Cost NAS with AZ-NAS ranking for MobileNet on CIFAR-10",
)
