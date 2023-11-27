from setuptools import setup,find_packages
import os 

here = os.path.abspath(os.path.dirname(__file__))
md_path = os.path.join(here, 'whyshift/README.md')

setup(
    name='whyshift',
    version='0.1.3',    
    description='A package of various specified distribution shift patterns of out-of-distributoin generalization problem on tabular data, and tools for diagnosing model performance are integrated.',
    long_description=open(md_path, encoding='utf-8').read(),
    long_description_content_type='text/markdown',
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