from setuptools import setup, find_packages

setup(
    name='dynlearn',
    version='0.1.0',
    # url='https://github.com/mypackage.git',
    author='Lorenz Wernisch',
    author_email='lwernisch@gmail.com',
    description='Dynamical system active learning with Gaussian processes',
    packages=find_packages(),
    install_requires=['scipy', 'matplotlib'],
    #
    # See https://github.com/tensorflow/tensorflow/issues/7166 for
    # rationale of TF requirement
    extras_require={
        "tf": ["tensorflow"],
        "tf_gpu": ["tensorflow-gpu"],
    }
)
