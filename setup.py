import setuptools

REQUIRED_PACKAGES = [
    "google-cloud-aiplatform==1.4.2",
    "tensorflow-transform==1.2.0",
    "tensorflow-data-validation==1.2.0",
]

setuptools.setup(
    name="executor",
    version="0.0.1",
    install_requires=REQUIRED_PACKAGES,
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={"src": ["raw_schema/schema.pbtxt"]},
)
