from setuptools import setup, find_packages

setup(
    name="quadrotor_diffusion",
    version="0.1",
    packages=find_packages(),
    install_requires=['numpy', 'torch', 'einops', 'scipy', 'matplotlib'],
)
