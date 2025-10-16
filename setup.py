"""
Setup configuration for ARP Spoofing Detection package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    requirements = [
        line.strip() 
        for line in requirements_file.read_text().split('\n') 
        if line.strip() and not line.startswith('#')
    ]
else:
    requirements = []

setup(
    name="arp_spoofing_detection",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-Based Real-Time ARP Spoofing Detection System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/arp-spoofing-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.3.0',
            'pytest-cov>=4.1.0',
            'black>=23.3.0',
            'flake8>=6.0.0',
            'mypy>=1.3.0',
        ],
        'docs': [
            'sphinx>=7.0.0',
            'sphinx-rtd-theme>=1.2.0',
        ],
        'deploy': [
            'flask>=2.3.0',
            'fastapi>=0.100.0',
            'uvicorn>=0.22.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'arp-train=scripts.train_model:main',
            'arp-detect=scripts.detect_realtime:main',
        ],
    },
    include_package_data=True,
    package_data={
        '': ['config/*.yaml', 'README.md', 'LICENSE'],
    },
)
