# Built on Folktables and the codebase https://github.com/jpgard/subgroup-robustness-grows-on-trees
__version__ = "0.1.3"
__author__ = 'Jiashuo Liu, Tianyu Wang, Peng Cui, Hongseok Namkoong'
__credits__ = 'Peng Cui\'s Group and Hongseok Namkoong\' Group'


from .dataset import get_data
from .algorithm import fetch_model
from .disde import degradation_decomp
from .region_analysis import risk_region