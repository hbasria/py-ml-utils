from setuptools import setup

setup(
    name="py-ml-utils",
    url="http://github.com/hbasria/py-ml-utils/",
    author="Hasan Basri",
    author_email="h@basri.me",
    version="0.1.0",
    packages=["ml_utils"],
    description="Machine learning utilities.",
    keywords=['Deep Learning', 'Machine Learning', 'Reinforcement Learning'],
    install_requires=['sklearn', 'xgboost'],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Environment :: Console",
        "Programming Language :: Python :: 3"
    ],
)
