from setuptools import setup, find_packages

setup(
    name="call-analytics",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "azure-ai-textanalytics>=5.3.0,<6.0.0",
        "bulk>=0.4.2,<0.5.0",
        "embetter[sentence-tfm]>=0.6.6,<0.7.0",
        "hiclass>=5.0.4,<6.0.0",
        "langdetect>=1.0.9,<2.0.0",
        "llvmlite>=0.44.0,<0.45.0",
        "nest-asyncio>=1.6.0,<2.0.0",
        "openai>=1.63.2,<2.0.0",
        "openpyxl>=3.1.5,<4.0.0",
        "pandas>=2.2.3,<3.0.0",
        "python-dotenv>=1.0.1,<2.0.0",
        "scikit-learn>=1.6.1,<2.0.0",
        "setfit>=1.1.1,<2.0.0",
        "torch>=2.6.0,<3.0.0",
        "torchaudio>=2.6.0,<3.0.0",
        "umap>=0.1.1,<0.2.0",
        "umap-learn>=0.5.7,<0.6.0",
    ],
    python_requires=">=3.10,<3.11",
) 