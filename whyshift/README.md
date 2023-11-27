## `WhyShift`: A Benchmark with Specified Distribution Shift Patterns 

> <a href="https://ljsthu.github.io">Jiashuo Liu*</a>, <a href="https://wangtianyu61.github.io">Tianyu Wang*</a>, <a href="https://pengcui.thumedialab.com">Peng Cui</a>, <a href="https://hsnamkoong.github.io">Hongseok Namkoong</a>

> Tsinghua University, Columbia University



`WhyShift` is a python package that provides a benchmark with various specified distribution shift patterns on real-world tabular data. And tools to diagnose performance degradation are integrated in it, including performance degradation decomposition and risky region identification. Our testbed highlights the importance of future research that builds an understanding of how distributions differ. For more details, please refer to our <a href="https://openreview.net/pdf?id=PF0lxayYST">paper</a>.

If you find this repository useful in your research, please cite the following paper:

```
@inproceedings{liu2023need,
  title={On the Need for a Language Describing Distribution Shifts: Illustrations on Tabular Datasets},
  author={Jiashuo Liu and Tianyu Wang and Peng Cui and Hongseok Namkoong},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2023}
}
```

### For settings utilizing ACS Income, Public Coverage, Mobility datasets
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


### For settings utilizing US Accident, Taxi datasets
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
