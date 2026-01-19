from setuptools import setup, find_packages

# Read requirements if available
try:
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
except FileNotFoundError:
    requirements = []

setup(
    name="fast-dllm",
    version="0.1.0.dev",
    description="Fast-DLLM: A diffusion-based Large Language Model inference acceleration framework (modified version)",
    packages=find_packages(include=["llada", "llada.*", "dream", "dream.*", "v2", "v2.*"]),
    python_requires=">=3.9",
    install_requires=requirements,
)
