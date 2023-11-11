[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?color=g&style=plastic)](https://opensource.org/licenses/MIT)
[![Downloads](https://static.pepy.tech/personalized-badge/whyshift?period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=Downloads)](https://pepy.tech/project/whyshift)
[![pypy: v](https://img.shields.io/pypi/v/whyshift.svg)](https://pypi.python.org/pypi/whyshift/)



## `WhyShift`: A Benchmark with Specified Distribution Shift Patterns 

> <a href="https://ljsthu.github.io">Jiashuo Liu*</a>, <a href="https://wangtianyu61.github.io">Tianyu Wang*</a>, <a href="https://pengcui.thumedialab.com">Peng Cui</a>, <a href="https://hsnamkoong.github.io">Hongseok Namkoong</a>

> Tsinghua University, Columbia University



`WhyShift` is a python package that provides a benchmark with various specified distribution shift patterns on real-world tabular data. Our testbed highlights the importance of future research that builds an understanding of how distributions differ. For more details, please refer to our <a href="https://openreview.net/pdf?id=PF0lxayYST">paper</a>.

If you find this repository useful in your research, please cite the following paper:

```
@inproceedings{liu2023need,
  title={On the Need for a Language Describing Distribution Shifts: Illustrations on Tabular Datasets},
  author={Jiashuo Liu and Tianyu Wang and Peng Cui and Hongseok Namkoong},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2023}
}
```


## Table of Contents
1. [Dataset Access](#dataset-access)
2. [Python Package: `whyshift`](#python-package-whyshift)
3. [Different Distribution Shift Patterns](#different-distribution-shift-patterns)
4. [Implemented Algorithms](#implemented-algorithms)
5. [Algorithm for Identifying Risk Region](#identify-risk-region)
6. [Degradation Decomposition (DISDE)](#disde-method)
7. [License and terms of use](#license-and-terms-of-use)
8. [References](#references)



## Dataset Access
Here we provide the access links for the 5 datasets used in our benchmark.

#### ACS Income
* The task is to predict whether an individual’s income is above \$50,000.
* Access link: https://github.com/socialfoundations/folktables
* Reference: Ding, F., Hardt, M., Miller, J., & Schmidt, L. (2021). Retiring adult: New datasets for fair machine learning. Advances in neural information processing systems, 34, 6478-6490.
* License: MIT License

#### ACS PubCov
* The task is to predict whether an individual has public health insurance.
* Access link: https://github.com/socialfoundations/folktables
* Reference: Ding, F., Hardt, M., Miller, J., & Schmidt, L. (2021). Retiring adult: New datasets for fair machine learning. Advances in neural information processing systems, 34, 6478-6490.
* License: MIT License

#### ACS Mobility
* The task is to predict whether an individual had the same residential address one year ago.
* Access link: https://github.com/socialfoundations/folktables
* Reference: Ding, F., Hardt, M., Miller, J., & Schmidt, L. (2021). Retiring adult: New datasets for fair machine learning. Advances in neural information processing systems, 34, 6478-6490.
* License: MIT License


#### Taxi Dataset
* The task is to predict whether the total ride duration time exceeds 30 minutes, based on location and temporal features.
* Access link: 
  * https://www.kaggle.com/datasets/mnavas/taxi-routes-for-mexico-city-and-quito
  * https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data    
* License: CC BY-SA 4.0


#### US Accident Dataset
* The task is to predict whether an accident is severe (long delay) or not (short delay) based on weather features and Road condition features.
* Access link: https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents
* License: CC BY-SA 4.0


## Python Package: `whyshift`
Here we provide the scripts to get data in our proposed settings.

#### Install the package
```
pip3 install whyshift
```

#### For settings utilizing ACS Income, Public Coverage, Mobility datasets
  * `get_data(task, state, year, need_preprocess, root_dir)` function
    * `task` values: 'income', 'pubcov', 'mobility'
  * examples:
    ```python
    from whyshift import get_data
    # for ACS Income
    X, y, feature_names = get_data("income", "CA", True, './datasets/acs/', 2018)
    # for ACS Public Coverage
    X, y, feature_names = get_data("pubcov", "CA", True, './datasets/acs/', 2018)
    # for ACS Mobility
    X, y, feature_names = get_data("mobility", "CA", True, './datasets/acs/', 2018)
    ```
  * support `state` values: 
    * ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']


#### For settings utilizing US Accident, Taxi datasets
  * download data files:
    ```python
    # US Accident:
    https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents
    # Taxi
    https://www.kaggle.com/competitions/nyc-taxi-trip-duration
    ```
  * put data files in dir `./datasets/`
    * accident: `./datasets/Accident/US_Accidents_Dec21_updated.csv`
    * taxi: `./datasets/Taxi/{city}_clean.csv`
  * pass the `path to the data file` of `get_data` function
  * example:
    ```python
    from whyshift import get_data
    # for US Accident
    X, y, _ = get_data("accident", "CA", True, './datasets/Accident/US_Accidents_Dec21_updated.csv')
    # for Taxi
    X, y, _ = get_data("taxi", "nyc", True, './datasets/Taxi/train.csv')
    ```
  * support `state` values:
    * for US Accident:  ['CA', 'TX', 'FL', 'OR', 'MN', 'VA', 'SC', 'NY', 'PA', 'NC', 'TN', 'MI', 'MO']
    * for Taxi: ['nyc', 'bog', 'uio', 'mex']


## Different Distribution Shift Patterns
Based on our `whyshift` package, one could design various source-target pairs with different distribution shift patterns. Here we list some of them for reference:

| #ID | Dataset | Type | #Features | Outcome | Source | #Train Samples | #Test Domains | Dom. Ratio |
| --- | ------- | ---- | --------- | ------- | ------ | -------------- | ------------- | ---------- |
| 1 | ACS Income | Spatial | 9 | Income≥50k | California | 195,665 | 50 | $Y\|X: 13/14$ |
| 2 | ACS Income | Spatial | 9 | Income≥50k | Connecticut | 19,785 | 50 | $Y\|X: 24/24$ |
| 3 | ACS Income | Spatial | 9 | Income≥50k | Massachusetts | 40,114 | 50 | $Y\|X: 21/22$ |
| 4 | ACS Income | Spatial | 9 | Income≥50k | South Dakota | 4,899 | 50 | $Y\|X: 9/9$ |
| 5 | ACS Mobility | Spatial | 21 | Residential Address | Mississippi | 5,318 | 50 | $Y\|X: 28/34$ |
| 6 | ACS Mobility | Spatial | 21 | Residential Address | New York | 40,463 | 50 | $Y\|X: 30/31$ |
| 7 | ACS Mobility | Spatial | 21 | Residential Address | California | 80,329 | 50 | $Y\|X: 9/17$ |
| 8 | ACS Mobility | Spatial | 21 | Residential Address | Pennsylvania | 23,918 | 50 | $Y\|X: 17/17$ |
| 9 | Taxi | Spatial | 7 | Duration time≥30 min | Bogotá | 3,063 | 3 | $Y\|X: 1/2$ |
| 10 | Taxi | Spatial | 7 | Duration time≥30 min | New York City | 1,458,646 | 3 | $Y\|X: 3/3$ |
| 11 | ACS Pub.Cov | Spatial | 18 | Public Ins. Coverage | Nebraska | 6,332 | 50 | $Y\|X: 32/39$ |
| 12 | ACS Pub.Cov | Spatial | 18 | Public Ins. Coverage | Florida | 71,297 | 50 | $Y\|X: 28/29$ |
| 13 | ACS Pub.Cov | Spatial | 18 | Public Ins. Coverage | Texas | 98,928 | 50 | $Y\|X: 33/34$ |
| 14 | ACS Pub.Cov | Spatial | 18 | Public Ins. Coverage | Indiana | 24,330 | 50 | $Y\|X: 11/13$ |
| 15 | US Accident | Spatial | 47 | Severity of Accident | Texas | 26,664 | 13 | $Y\|X: 7/7$ |
| 16 | US Accident | Spatial | 47 | Severity of Accident | California | 64,909 | 13 | X: 22/31 |
| 17 | US Accident | Spatial | 47 | Severity of Accident | Florida | 32,278 | 13 | X: 5/7 |
| 18 | US Accident | Spatial | 47 | Severity of Accident | Minnesota | 8,927 | 13 | X: 8/11 |
| 19 | ACS Pub.Cov | Temporal | 18 | Public Ins. Coverage | Year 2010 (NY) | 73,208 | 3 | X: 2/2 |
| 20 | ACS Pub.Cov | Temporal | 18 | Public Ins. Coverage | Year 2010 (CA) | 149,441 | 3 | X: 2/2 |
| 21 | ACS Income | Synthetic | 9 | Income≥50k | Younger People (80%) | 20,000 | 1 | X: 1/1 |
| 22 | ACS Income | Synthetic | 9 | Income≥50k | Younger People (90%) | 20,000 | 1 | X: 1/1 |

In our benchmark, each setting has multiple target domains (except the last setting). In our main body, we select only one target domain for each setting. We report the `Dom. Ratio` to represent the dominant ratio of $Y|X$ shifts or $X$ shifts in source-target pairs with performance degradation larger than **5** percentage points in each setting. For example, "$Y|X$: 13/14" means that there are 14 source-target pairs in Setting 1 with degradation larger than 5 percentage points and 13 out of them with over 50\% degradation attributed to $Y|X$ shifts. We use XGBoost to measure this.


## Implemented Algorithms
In our `whyshift` package, we also implement several algorithms for tabular data classification, including `Logistic Regression`, `MLP`, `SVM`, `Random Forest`, `XGBoost`, `LightGBM`, `GBM`, $\chi^2$/CVaR-`DRO/DORO`, `Group DRO`, `Simple-Reweighting`, `JTT`, `Fairness-In/Postprocess` and `DWR` methods.

```python
# use the implemented methods
algo = fetch_model(method_name)
```

Note that the supported method names are:

```python
method_name_list = ['lr','svm','xgb', 'lightgbm', 'rf',  'dwr', 'jtt','suby', 'subg', 'rwy', 'rwg', 'FairPostprocess_exp','FairInprocess_dp', 'FairPostprocess_threshold', 'FairInprocess_eo', 'FairInprocess_error_parity','chi_dro', 'chi_doro','cvar_dro','cvar_doro','group_dro']
```

## Identify Risk Region
In our `whyshift` package, we implement the risk region identification algorithm (Algorithm 1 in our paper). The function is `risk_region`.
And here is an example to use it:

```python
from whyshift import risk_region

source_model = xgb.XGBClassifier()
target_model = xgb.XGBClassifier()
risk_region('xgb', source_model, target_model, 'income', 'CA', 'PR', "./datasets/acs")
```



## DISDE Method
In our `whyshift` package, we implement the DIstribution Shift DEcomposition (`DISDE`) method to attribute the performance degradation to $Y|X$-shifts and $X$-shifts, respectively. 
Function `degradation_decomp` 

The parameters include: 
* source_X, source_y, other_X_raw, other_y_raw: data from the source and target distributions
* best_method: the model to be diagnosed
* data_sum: the sample num of target data
* K: K-fold training-testing
* domain_classifier: when not specified (None), XGBoost will be used
* draw_calibration: whether to draw the calibration curve to check the quality of domain classifier
* save_calibration_png: the path to save the calibration figure

An example to use our `WHYSHIFT` package could be found at <a href="./Example.ipynb">`Example.ipynb`</a>.


## License and terms of use
Our benchmark is built upon `Folktables`. The License of `Folktables` is:

```
Folktables provides code to download data from the American Community Survey (ACS) Public Use Microdata Sample (PUMS) files managed by the US Census Bureau. The data itself is governed by the terms of use provided by the Census Bureau. For more information, see https://www.census.gov/data/developers/about/terms-of-service.html

The Adult reconstruction dataset is a subsample of the IPUMS CPS data available from https://cps.ipums.org/. The data are intended for replication purposes only. Individuals analyzing the data for other purposes must submit a separate data extract request directly via IPUMS CPS. Individuals are not to redistribute the data without permission. Contact ipums@umn.edu for redistribution requests.
```
Besides, for US Accident and Taxi data from `kaggle`, individuals should follow the their Licenses, see https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents and https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data.

## References
[1] Ding, F., Hardt, M., Miller, J., & Schmidt, L. (2021). Retiring adult: New datasets for fair machine learning. Advances in neural information processing systems, 34, 6478-6490.

* We modify the <a href="https://github.com/socialfoundations/folktables">`folktables`</a> code to support `year` before 2014, and involve the revised version in our package. 
* Part of the algorithm codes are used from the <a href="https://github.com/jpgard/subgroup-robustness-grows-on-trees">codebase</a>.





<!-- if __name__ == '__main__':
    source_model = xgb.XGBClassifier()
    target_model = xgb.XGBClassifier()
    
    compare_best_model('xgb', source_model, target_model, 'income', 'CA', 'PR', "/home/jiashuoliu/TabularWilds/src/datasets/acs")
  -->
