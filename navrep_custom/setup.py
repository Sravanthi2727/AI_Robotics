from setuptools import setup, find_packages

setup(
    name="navrep_custom",
    version="1.0.0",
    description="Custom benchmark scenarios and environments for ETH NavRep",
    author="Robotics-ML Engineer",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "gym==0.15.7",
        "opencv-python",
        "matplotlib",
        "pandas"
    ],
)

