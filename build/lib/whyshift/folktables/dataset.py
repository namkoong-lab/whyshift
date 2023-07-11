import numpy as np
import pickle
from acs import ACSDataSource, ACSEmployment, ACSIncome, ACSPublicCoverage, ACSMobility, ACSTravelTime
from folktables import BasicProblem
from folktables import *
import pandas as pd
import random
import seaborn as sns
import pickle





def get_ACSIncome(state, year=2018):
    task = ACSIncome
    data_source = ACSDataSource(root_dir='../', survey_year=year, horizon='1-Year', survey='person')
    source_data = data_source.get_data(states=[state], download=True)
    





if __name__ == "__main__":
    for state in ['CA', 'TX', 'FL', 'NY', 'MT', 'SD']:
        for year in [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2019, 2021]:
            get_ACSIncome(state, year)
    # get_USAccident('xgb', 'CA')