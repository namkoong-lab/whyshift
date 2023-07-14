from setuptools import setup,find_packages

setup(
    name='whyshift',
    version='0.0.4',    
    description='A package of various specified distribution shift patterns of out-of-distributoin generalization problem on tabular data',
    url='https://github.com/namkoong-lab/whyshift',
    author='Jiashuo Liu, Tianyu Wang, Peng Cui, Hongseok Namkoong',
    author_email='liujiashuo77@gmail.com, tw2837@columbia.edu, cuip@tsinghua.edu.cn, namkoong@gsb.columbia.edu',
    packages=find_packages(),
    install_requires=['pandas',
                      'numpy',                     
                      'scikit-learn',
                      'lightgbm','xgboost','fairlearn', 'tqdm', 'torch', 'scipy'
                      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
        'Operating System :: POSIX :: Linux',
    ],
    python_requires=">=3",
)