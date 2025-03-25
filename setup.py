from setuptools import setup, find_packages

setup(
    name='melimda',
    version='0.0.1',
    description='melimda: machine-learning improved docking algorithm',
    long_description='melimda: machine-learning improved docking algorithm',
    long_description_content_type='text/markdown',
    author='Lucas Mocellin Goulart; Frederico Schmitt Kremer',
    author_email='lmocellingoulart@gmail.com',
    packages=find_packages(),
    install_requires=['deepchem==2.7.1', 'pandas', 'numpy', 'rdkit-pypi', 'tensorflow', 'jax', 'torch', 'torch-geometric'],
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'melimda-preprocess = melimda.preprocess:main',
            'melimda-train = melimda.train:main',
            'melimda-predict = melimda.predict:main',
        ]
    }
)
