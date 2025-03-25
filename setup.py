# setup.py
from setuptools import setup, find_packages

setup(
    name="auto_trainer",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "optuna",
        "datasets",
        "numpy",
        "scikit-learn"
    ],
    author="Xinyuan Tan, Zhouzhao Xu, Zhuoyuan Chen",
    description="Auto training and NAS proxy-based optimizer for CNN/Transformer models",
    url="https://github.com/xzz17/Advanced-AZ-NAS",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
