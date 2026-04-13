from setuptools import setup, find_packages

setup(
    name="databricks_agentic_tune",
    version="1.0.0",
    author="vijayakunuri1",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "run-qc-pipeline=pipelines.full_qc_pipeline:main",
        ]
    },
    python_requires=">=3.10",
)
