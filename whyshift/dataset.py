import numpy as np
from . import folktables_utils
from .folktables import ACSDataSource, ACSEmployment, ACSIncome, ACSPublicCoverage, ACSMobility, ACSTravelTime
from .folktables import BasicProblem
import pandas as pd
from sklearn import preprocessing
import pickle
from sklearn.preprocessing import StandardScaler
from .folktables_utils import add_indicators, add_indicators_year, add_indicators_pubcov, add_indicators_traveltime, add_indicators_mobility
from .utils import preprocess

rac1p_vals = ['white','black','am_ind','alaska','am_alaska','asian','hawaiian','other','two_or_more']
relp_vals = ['reference', 'husband/wife','biologicalson','adoptedson','stepson','brother','father','grandchild','parentinlaw','soninlaw','other','roomer',
    'housemate','unmarried','foster','nonrelative','institutionalized','noninstitutionalized']
SCHL_vals = ['SCHL', 'schl_at_least_bachelor', 'schl_at_least_high_school_or_ged', 'schl_postgrad']
COW_vals = ['cow_employee_profit', 'cow_employee_nonprofit', 'cow_localgovernment', 'cow_stategovernment', 'cow_federalgovernment', 'cow_selfemployed_own',\
            'cow_selfemployed_incorporated', 'cow_family_business', 'cow_unemployed']
OCCP_vals = ['MGR1', 'MGR2', 'MGR3', 'MGR4', 'MGR5', 'BUS1', 'BUS2', 'BUS3', 'FIN1', 'FIN2', 
             'CMM1', 'CMM2', 'CMM3', 'ENG1', 'ENG2', 'ENG3', 'SCI1', 'SCI2', 'SCI3', 'SCI4', 
             'CMS1', 'LGL1', 'EDU1', 'EDU2', 'EDU3', 'EDU4', 'ENT1', 'ENT2', 'ENT3', 'ENT4', 
             'MED1', 'MED2', 'MED3', 'MED4', 'MED5', 'MED6', 'HLS1', 'PRT1', 'PRT2', 'PRT3',  
             'EAT1', 'EAT2', 'CLN1', 'PRS1', 'PRS2', 'PRS3', 'PRS4', 'SAL1', 'SAL2', 'SAL3', 
             'OFF1', 'OFF2', 'OFF3', 'OFF4', 'OFF5', 'OFF6', 'OFF7', 'OFF8', 'OFF9', 'OFF10', 
             'FFF1', 'FFF2', 'CON1', 'CON2', 'CON3', 'CON4', 'CON5', 'CON6', 'EXT1', 'EXT2', 
             'RPR1', 'RPR2', 'RPR3', 'RPR4', 'RPR5', 'RPR6', 'RPR7', 'PRD1', 'PRD2', 'PRD3', 
             'PRD4', 'PRD5', 'PRD6', 'PRD7', 'PRD8', 'PRD9', 'PRD10', 'PRD11', 'PRD12', 'PRD13', 
             'TRN1', 'TRN2', 'TRN3', 'TRN4', 'TRN5', 'TRN6', 'TRN7', 'TRN8', 'MIL1', "no1"]
Big_OCCP_vals = ['MGR', 'BUS', 'FIN', 'CMM', 'ENG', 'SCI', 'CMS', 'EDU', 'ENT', 'MED', 'HLS', 'PRT', 'EAT', 'CLN', 'PRS',\
                'SAL', 'OFF', 'FFF', 'CON', 'EXT', 'RPR', 'PRD', 'TRN', 'MIL', 'no']
Large_OCCP_vals = ['Lg', 'Semi_Lg', 'Non_Lg']
CIT_vals = ['us', 'pr', 'abroad', 'citizen', 'not']
ESR_vals = ['employed', 'partial_employed', 'unemployed', 'armed', 'partial_armed', 'no']


METHOD_NONE_DUMMIES = []
METHOD_NEED_PREPROCESS = ['svm', 'chi_doro', 'cvar_doro']

def adult_filter(data):
    """Mimic the filters in place for Adult data.
    Adult documentation notes: Extraction was done by Barry Becker from
    the 1994 Census database. A set of reasonably clean records was extracted
    using the following conditions:
    ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))
    """
    df = data
    df = df[df['AGEP'] > 16]
    df = df[df['PINCP'] > 100]
    df = df[df['WKHP'] > 0]
    df = df[df['PWGTP'] >= 1]
    return df
def travel_time_filter(data):
    """
    Filters for the employment prediction task
    """
    df = data
    df = df[df['AGEP'] > 16]
    df = df[df['PWGTP'] >= 1]
    df = df[df['ESR'] == 1]
    return 

def employment_filter(data):
    """
    Filters for the employment prediction task
    """
    df = data
    df = df[df['AGEP'] > 16]
    df = df[df['AGEP'] < 90]
    df = df[df['PWGTP'] >= 1]
    return 

def public_coverage_filter(data):
    """
    Filters for the public health insurance prediction task; focus on low income Americans, and those not eligible for Medicare
    """
    df = data
    df = df[df['AGEP'] < 65]
    df = df[df['PINCP'] <= 30000]
    return df

def mobility_filter(data):
    df = data 
    df = df[df["AGEP"] > 18]
    df = df[df["AGEP"] < 35]
    return df

def get_USAccident(state, need_preprocess, root_dir='./datasets/USAccident/US_Accidents_Dec21_updated.csv'):
    if not state in ['CA', 'TX', 'FL', 'OR', 'MN', 'VA', 'SC', 'NY', 'PA', 'NC', 'TN', 'MI', 'MO']:
        raise NotImplementedError(f"{state} is not supported in this dataset!")

    raw_X = preprocess(root_dir)
    data = raw_X[raw_X["State"]==state]

    y_sample = data["Severity"]
    X_sample = data.drop(["Severity", "State", "Start_Lat", "Start_Lng"], axis=1).values

    if need_preprocess:
        scaler = StandardScaler()
        scaler.fit(X_sample)
        X_sample = scaler.transform(X_sample)

    # print(X_sample.shape, yq_sample.shape)
    return X_sample, y_sample.to_numpy().astype('int'), None

def get_taxi(city, need_preprocess,root_dir='./datasets/taxi/nyc_clean.csv'):

    remove_col_nyc = ['id', 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'passenger_count']
    remove_col_other = ['id', 'dist_meters', 'wait_sec', 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']

    try:
        df = pd.read_csv(root_dir)
    except:
        raise FileNotFoundError('File does not exist: {}'.format(root_dir))
        
    df = df[(df.trip_duration < 5900)]
    df = df[(df.pickup_longitude > -110)]
    df = df[(df.pickup_latitude < 50)]
    df.drop(['store_and_fwd_flag'], axis=1, inplace=True)
    df.drop(['vendor_id'], axis=1, inplace=True)
    df['pickup_datetime'] = pd.to_datetime(df.pickup_datetime)
    df.drop(['dropoff_datetime'], axis=1, inplace=True) 
    df['month'] = df.pickup_datetime.dt.month
    df['week'] = df.pickup_datetime.dt.isocalendar().week
    df['weekday'] = df.pickup_datetime.dt.weekday
    df['hour'] = df.pickup_datetime.dt.hour
    df['minute'] = df.pickup_datetime.dt.minute
    df['minute_oftheday'] = df['hour'] * 60 + df['minute']
    df.drop(['minute'], axis=1, inplace=True)


    df.drop(['pickup_datetime'], axis=1, inplace=True)

    def ft_haversine_distance(lat1, lng1, lat2, lng2):
        lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
        AVG_EARTH_RADIUS = 6371 #km
        lat = lat2 - lat1
        lng = lng2 - lng1
        d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
        h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
        return h

    df['distance'] = ft_haversine_distance(df['pickup_latitude'].values,
                                                    df['pickup_longitude'].values, 
                                                    df['dropoff_latitude'].values,
                                                    df['dropoff_longitude'].values)

    def ft_degree(lat1, lng1, lat2, lng2):
        AVG_EARTH_RADIUS = 6371 #km
        lng_delta_rad = np.radians(lng2 - lng1)
        lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
        y = np.sin(lng_delta_rad) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
        return np.degrees(np.arctan2(y, x))

    df['direction'] = ft_degree(df['pickup_latitude'].values,
                                    df['pickup_longitude'].values,
                                    df['dropoff_latitude'].values,
                                    df['dropoff_longitude'].values)


    df = df[(df.distance < 200)]
    df['speed'] = df.distance / df.trip_duration
    df = df[(df.speed < 30)]

    df.drop(['speed'], axis=1, inplace=True)

    try:
        df = df.sample(n = 10000)
        test = test.sample(n = 10000)
    except:
        pass

    y_sample = df["trip_duration"].apply(lambda x: 0 if x < 900 else 1)
    df.drop(["trip_duration"], axis=1, inplace=True)
    if city == 'nyc':
        df.drop(remove_col_nyc, axis=1, inplace=True)
    else:
        df.drop(remove_col_other, axis = 1, inplace = True)

    X_sample = df.to_numpy()

    if need_preprocess:
        scaler = StandardScaler()
        scaler.fit(X_sample)
        X_sample = scaler.transform(X_sample)

    return X_sample, y_sample.to_numpy().astype('int'), None

def get_ACSIncome(state, year=2018, need_preprocess=True, root_dir = './datasets/acs'):
    task = ACSIncome
    data_source = ACSDataSource(root_dir=root_dir, survey_year=year, horizon='1-Year', survey='person')
    source_data = data_source.get_data(states=[state], download=True)
    
    source_data = adult_filter(source_data)
    
    new_features = ['SEX', 'AGEP', 'WKHP',  'married', 'widowed','divorced','separated','never']+['relp_'+x for x in relp_vals]\
                    +['race_'+x for x in rac1p_vals] + COW_vals +['big_occp_'+x for x in Big_OCCP_vals]\
                    +SCHL_vals+['large_occp_'+x for x in Large_OCCP_vals]
    
    new_task = BasicProblem(new_features, task._target, task._target_transform,
                task._group, task._group_transform,
                preprocess = add_indicators, postprocess = task._postprocess)

    source_X_raw, source_y_raw, _ = new_task.df_to_numpy(source_data)
    
    if need_preprocess:
        scaler = StandardScaler()
        scaler.fit(source_X_raw[:,2:])
        source_X_raw[:,2:] = scaler.transform(source_X_raw[:,2:])
        
    return source_X_raw, source_y_raw.astype("int"), new_features

def get_ACSIncome_aug(state, year=2018, need_preprocess=True, root_dir = './datasets/acs'):
    """
    Add features to mitigate the concept drifts
    """
    task = ACSIncome
    data_source = ACSDataSource(root_dir=root_dir, survey_year=year, horizon='1-Year', survey='person')
    source_data = data_source.get_data(states=[state], download=True)
    source_data = adult_filter(source_data)
    
    new_features = ['SEX', 'AGEP', 'WKHP',  'married', 'widowed','divorced','separated','never']+['relp_'+x for x in relp_vals]\
                    +['race_'+x for x in rac1p_vals] + COW_vals +['big_occp_'+x for x in Big_OCCP_vals]\
                    +SCHL_vals+['large_occp_'+x for x in Large_OCCP_vals]+['ENG']
    
    new_task = BasicProblem(new_features, task._target, task._target_transform,
                task._group, task._group_transform,
                preprocess = add_indicators, postprocess = task._postprocess)

    source_X_raw, source_y_raw, _ = new_task.df_to_numpy(source_data)
    
    index = np.where(source_X_raw[:,-1]>2)[0]
    if need_preprocess:
        scaler = StandardScaler()
        scaler.fit(source_X_raw[:,1:])
        source_X_raw[:,1:] = scaler.transform(source_X_raw[:,1:])

    return source_X_raw, source_y_raw.astype("int"), new_features

def get_ACSPubCov(state, year=2018, need_preprocess=True, root_dir = './datasets/acs'):
    task = ACSPublicCoverage
    data_source = ACSDataSource(root_dir=root_dir, survey_year=year, horizon='1-Year', survey='person')
    source_data = data_source.get_data(states=[state], download=False)
    
    source_data = public_coverage_filter(source_data)
    rac1p_vals = ['white','black','am_ind','alaska','am_alaska','asian','hawaiian','other','two_or_more']
    
    new_features = ['SEX', 'AGEP', 'DIS', 'ESP', 'MIG', 'MIL', 'ANC', 'NATIVITY', 'DEAR', 'DEYE',
                    'DREM', 'PINCP', 'FER',  'married', 'widowed','divorced','separated','never']+['race_'+x for x in rac1p_vals]+SCHL_vals+['CIT_'+x for x in CIT_vals]\
                    +['ESR_'+x for x in ESR_vals]
    
    new_task = BasicProblem(new_features, task._target, task._target_transform,
                task._group, task._group_transform,
                preprocess = add_indicators_pubcov, postprocess = task._postprocess)

    source_X_raw, source_y_raw, _ = new_task.df_to_numpy(source_data)
    
    if need_preprocess:
        scaler = StandardScaler()
        scaler.fit(source_X_raw[:,2:])
        source_X_raw[:,2:] = scaler.transform(source_X_raw[:,2:])
        
    return source_X_raw, source_y_raw.astype("int"), new_features

def get_ACSMobility(state, year=2018, need_preprocess=True, root_dir = './datasets/acs'):
    task = ACSMobility
    data_source = ACSDataSource(root_dir=root_dir, survey_year=year, horizon='1-Year', survey='person')
    source_data = data_source.get_data(states=[state], download=False)
    
    source_data = mobility_filter(source_data)
    rac1p_vals = ['white','black','am_ind','alaska','am_alaska','asian','hawaiian','other','two_or_more']
    relp_vals = ['reference', 'husband/wife','biologicalson','adoptedson','stepson','brother','father','grandchild','parentinlaw','soninlaw','other','roomer',
        'housemate','unmarried','foster','nonrelative','institutionalized','noninstitutionalized']
    new_features = ['SEX', 'AGEP', 'SCHL', 'DIS', 'ESP','CIT', 'MIL', 'ANC', 'NATIVITY',  'DEAR','DEYE','DREM',
                    'GCL',  'WKHP', 'JWMNP', 'PINCP', 'married', 'widowed','divorced','separated','never']\
                    +['race_'+x for x in rac1p_vals]+['relp_'+x for x in relp_vals]+COW_vals+['ESR_'+x for x in ESR_vals]
    
    new_task = BasicProblem(new_features, task._target, task._target_transform,
                task._group, task._group_transform,
                preprocess = add_indicators_mobility, postprocess = task._postprocess)

    source_X_raw, source_y_raw, _ = new_task.df_to_numpy(source_data)
    
    if need_preprocess:
        scaler = StandardScaler()
        scaler.fit(source_X_raw[:,2:])
        source_X_raw[:,2:] = scaler.transform(source_X_raw[:,2:])
        
    return source_X_raw, source_y_raw.astype("int"), new_features


def get_data(task, state, need_preprocess, root_dir, year=2018):
    if task == 'income':
        return get_ACSIncome(state, year, need_preprocess, root_dir)
    elif task == 'income_aug':
        return get_ACSIncome_aug(state, year, need_preprocess, root_dir)
    elif task == 'pubcov':
        return get_ACSPubCov(state, year, need_preprocess, root_dir)
    elif task == 'mobility':
        return get_ACSMobility(state, year, need_preprocess, root_dir)
    elif task == 'accident':
        return get_USAccident(state, need_preprocess, root_dir)
    elif task == 'taxi':
        return get_taxi(state, need_preprocess, root_dir)
