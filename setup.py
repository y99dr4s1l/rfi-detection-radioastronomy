from setuptools import setup, find_packages

setup(
    name="rfi-detection-radioastronomy",
    version="0.1.0",
    description="Comparison of statistical and machine learning methods for RFI detection in radio astronomical spectrograms",
    author="",
    python_requires=">=3.9",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.23,<2.0",
        "pandas>=1.5",
        "scipy>=1.10",
        "tqdm>=4.65",
        "scikit-learn>=1.3",
        "scikit-image>=0.21",
        "matplotlib>=3.7",
        "tables>=3.8",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "jupyter>=1.0",
        ]
    },
)