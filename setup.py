import setuptools

REQUIRED_PACKAGES = [
    "google-cloud-aiplatform==1.0.0",
    "tensorflow-transform==0.30.0",
    "tensorflow-data-validation==0.30.0",
]

setuptools.setup(
    name="executor",
    version="0.0.1",
    install_requires=REQUIRED_PACKAGES,
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={"src": ["raw_schema/schema.pbtxt"]},
)
