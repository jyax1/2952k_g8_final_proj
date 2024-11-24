from setuptools import setup, find_packages

setup(
    name="action_extractor",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "h5py",
        "zarr",
        "einops",
        "torchvideotransforms",
        "tqdm",
        "matplotlib",
        "seaborn"
    ],
    package_data={
        'action_extractor': ['utils/*.npz']
    },
    include_package_data=True
)