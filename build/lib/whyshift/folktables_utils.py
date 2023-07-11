import pandas as pd
import numpy as np
import functools
from sklearn.model_selection import train_test_split
from .folktables import adult_filter

def flatten_list(list_of_lists):
        return functools.reduce(lambda x,y:x+y, list_of_lists)

rs = 0
s = 'AL,AK,AZ,AR,CA,CO,CT,DE,FL,GA,HI,ID,IL,IN,IA,KS,KY,LA,ME,MD,MA,MI,MN,MS,MO,MT,NE,NV,NH,NJ,NM,NY,NC,ND,OH,OK,OR,PA,RI,SC,SD,TN,TX,UT,VT,VA,WA,WV,WI,WY'
all_states = s.split(',')
rac1p_vals = ['white','black','am_ind','alaska','am_alaska','asian','hawaiian','other','two_or_more']
feature_categories = ['AGEP','RACE','SCHL','MAR','MIL','CIT','MIG','DIS','SEX','NATIVITY']
relp_vals = ['reference', 'husband/wife','biologicalson','adoptedson','stepson','brother','father','grandchild','parentinlaw','soninlaw','other','roomer',
            'housemate','unmarried','foster','nonrelative','institutionalized','noninstitutionalized']
jwtr_vals = ['car', 'bus', 'streetcar', 'subway', 'railroad', 'ferryboat', 'taxicab', 'motocycle', 'bicycle', 'walk', 'home', 'other']
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

Big_OCCP_vals = ['MGR', 'BUS', 'FIN', 'CMM', 'ENG', 'SCI', 'CMS', 'LGL', 'EDU', 'ENT', 'MED', 'HLS', 'PRT', 'EAT', 'CLN', 'PRS',
                'SAL', 'OFF', 'FFF', 'CON', 'EXT', 'RPR', 'PRD', 'TRN', 'MIL', 'no']
Big_OCCP_threshold = [[0,4], [5,7], [8,9], [10,12], [13,15], [16,19], [20, 20], [21, 21], [22, 25],\
                    [26, 29], [30,35], [36,36], [37,39], [40,41], [42, 42], [43,46], [47,49], [50,59],\
                    [60,61], [62,67], [68,69], [70,76], [77,89], [90,97], [98,98],[99,99]]
Large_OCCP_vals = ['Lg', 'Semi_Lg', 'Non_Lg']
CIT_vals = ['us', 'pr', 'abroad', 'citizen', 'not']
ESR_vals = ['employed', 'partial_employed', 'unemployed', 'armed', 'partial_armed', 'no']

def add_esr_indicators(t):
    for idx, esr in enumerate(ESR_vals):
        t['ESR_%s'%esr] = t.ESR == (idx+1)

def add_cit_indicators(t):
    for idx, cit in enumerate(CIT_vals):
        t['CIT_%s'%cit] = t.CIT == (idx+1)

def add_occp_indicators(t):
    for idx, occp in enumerate(OCCP_vals):
        t['occp_%s'%occp] = t.OCCP == idx
    return t

def add_big_OCCP_indicators(t):
    for idx, big_occp in enumerate(Big_OCCP_vals):
        t['big_occp_%s'%big_occp] = ((t.OCCP >= Big_OCCP_threshold[idx][0]) & (t.OCCP <= Big_OCCP_threshold[idx][1]))
    return t 

def add_large_occp_indicator(t):
    t['large_occp_Lg'] = ((t.big_occp_MGR == 1)|(t.big_occp_BUS == 1)|(t.big_occp_FIN == 1)|\
                          (t.big_occp_LGL == 1)|(t.big_occp_EDU == 1)|(t.big_occp_ENT == 1))

    t['large_occp_Semi_Lg'] = ((t.big_occp_CMM == 1)|(t.big_occp_ENG == 1)|(t.big_occp_SCI == 1)|\
                          (t.big_occp_CMS == 1)|(t.big_occp_MED == 1)|(t.big_occp_HLS == 1)|(t.big_occp_PRS == 1)|\
                          (t.big_occp_SAL == 1)|(t.big_occp_OFF == 1)|(t.big_occp_RPR == 1)|(t.big_occp_PRD == 1))

    t['large_occp_Non_Lg'] = ((t.big_occp_PRT == 1)|(t.big_occp_EAT == 1)|(t.big_occp_CLN == 1)|\
                          (t.big_occp_FFF == 1)|(t.big_occp_CON == 1)|(t.big_occp_EXT == 1)|(t.big_occp_TRN == 1)\
                          |(t.big_occp_MIL == 1)|(t.big_occp_no == 1))


def add_race_indicators(t):
    for idx, race in enumerate(rac1p_vals):
        t['race_%s'%race] = t.RAC1P == (idx+1)
    return t
def add_relp_indicators(t):
    for idx, relp in enumerate(relp_vals):
        t['relp_%s'%relp] = t.RELP == idx
    return t

def add_jwtr_indicators(t):
    for idx, relp in enumerate(jwtr_vals):
        t['jwtr_%s'%relp] = t.JWTR == (idx+1)
    return t

def add_school_indicators(t):
    t['schl_at_least_bachelor']=t.SCHL >= 21
    t['schl_at_least_high_school_or_ged']=t.SCHL >= 17
    t['schl_postgrad']=t.SCHL >= 22
    return t
def add_married_indicator(t):
    t['married']=t.MAR==1
    t['widowed']=t.MAR==2
    t['divorced']=t.MAR==3
    t['separated']=t.MAR==4
    t['never']=t.MAR==5
    return t
def add_military_indicators(t):
    t['active_military']=t.MIL==1
    t['vet']=t.MIL==2
    return t
def add_citizenship_indicators(t):
    t['cit_born_us']=t.CIT==1
    t['cit_born_territory']=t.CIT==2
    t['cit_am_parents']=t.CIT==3
    t['cit_naturalized']=t.CIT==4
    t['cit_not_citizen']=t.CIT==5
    return t
def add_mobility_indicators(t):
    t['mig_moved']=t.MIG!=1
    return t

def add_cow_indicators(t):
    for idx, cow in enumerate(COW_vals):
        t[cow] = t.COW == (idx+1)
    return t
        


def add_indicators(t):
    adult_filter(t)
    add_race_indicators(t)
    add_relp_indicators(t)
    add_school_indicators(t)
    add_married_indicator(t)
    t.OCCP = t.OCCP // 100
    add_big_OCCP_indicators(t)
    # add_occp_indicators(t)
    add_large_occp_indicator(t)
    add_cow_indicators(t)
    return t

def add_indicators_year(t):
    adult_filter(t)
    add_race_indicators(t)
    # add_relp_indicators(t)
    add_school_indicators(t)
    add_married_indicator(t)
    
    # add_military_indicators(t)
    # add_citizenship_indicators(t)
    # add_mobility_indicators(t)
    return t

def add_indicators_pubcov(t):
    add_race_indicators(t)
    add_school_indicators(t)
    add_married_indicator(t)
    add_cit_indicators(t)
    add_esr_indicators(t)
    return t

def add_indicators_traveltime(t):
    add_race_indicators(t)
    add_school_indicators(t)
    add_married_indicator(t)
    add_jwtr_indicators(t)
    return t

def add_indicators_mobility(t):
    add_race_indicators(t)
    add_school_indicators(t)
    add_married_indicator(t)
    add_relp_indicators(t)
    add_cow_indicators(t)
    add_esr_indicators(t)
    return t

binarized_features = ['race_'+x for x in rac1p_vals] + ['married',
                                                    'schl_at_least_bachelor',
                                                    'schl_at_least_high_school_or_ged',
                                                    'schl_postgrad',
                                                    'active_military','vet',
                                                    'cit_born_us',
                                                    'cit_born_territory',
                                                    'cit_am_parents',
                                                    'cit_naturalized',
                                                    'cit_not_citizen','mig_moved']
