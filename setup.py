import setuptools

setuptools.setup(
    name="methBERT",
    version="1.0",
    author="Yao-zhong Zhang",
    author_email="yaozhong@ims.u-tokyo.ac.jp",
    description="BERT model applied for nanopore methyaltion detection",
    long_description="",
    long_description_content_type="",
    url="",
    packages=setuptools.find_packages(),
    package_data={'methBERT':['methBERT/*'],'methBERT.dataProcess':['methBERT/dataProcess/*'],'methBERT.model':['methBERT/model/*']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['numpy', 'seaborn', 'matplotlib','seaborn','statsmodels','h5py','tqdm','sklearn'],
)
