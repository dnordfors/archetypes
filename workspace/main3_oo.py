

#%%

# path to working directory
path = '/Users/davidnordfors/galvanize/galvanize-capstone/final'


# IMPORT LIBRARIES
## OS
import os
os.chdir(path)

# MANAGE
import pandas as pd
import numpy as np

## FIT
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb

#from sklearn.metrics import r2_score

## DECOMPOSITION
from sklearn.decomposition import NMF
from scipy.linalg import svd

## GRAPHICS
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})

## I/O
import zipfile
import requests
import pickle

## FUNCTIONS AND CLASSES

# Normalization function
def norm(vec):
    '''
    Normalizes a vector v-v.mean())/v.std() - works for: PandasDataFrame.apply(norm)
    '''
    return (vec-vec.mean())/vec.std()

def nrmcol(df):
    '''
    Normalizes the columns of a DataFrame (cos distance)
    '''
    return df / np.sqrt(np.diagonal(df.T @ df))

# READ FROM O*NET DATABASE. 
onet = {}
def from_onet(qualities):
    '''
    READ O*NET DATABASE. Search order: Dictionary, Pickle, Excel; Create dictionary/pickle if non-existent.
    '''
    if qualities in onet:
            return onet[qualities]
    pickle_exists = os.path.isfile('./data/pickle/'+qualities+'.pkl')
    if pickle_exists:
            onet[qualities] = pd.read_pickle('./data/pickle/'+qualities+'.pkl')
            return onet[qualities]
    else: 
            onet[qualities]= pd.read_excel(
                    zipfile.ZipFile('./data/db_23_2_excel.zip').extract(
                    'db_23_2_excel/'+qualities+'.xlsx'))
            onet[qualities].to_pickle('./data/pickle/'+qualities+'.pkl')
            return onet[qualities]


# format and Strip O*NET SOC occupation codes to match the ones used by Census PUMS
def soc(socpnr,shave=5):
    '''
    Key function 
        1. formats SOC-occupation codes – for matching O*Net and census.
        2. 'Shaves' the SOC-number to length 'shave'. The number sets the granularity of job classifications. 
    '''
    socpnr2 = str(socpnr).replace('-','')
    socp_shave = socpnr2[:shave]
    return socp_shave

# CENSUS DATA: 
# PUMS Data dictionary
#Source: https://www.census.gov/programs-surveys/acs/data/pums.html )
datadic = pd.read_csv("./data/PUMS_Data_Dictionary_2017.csv").drop_duplicates()

# rows including the string 'word'
def var_about(word):
    return pd.concat((BM(datadic).select('Record Type', 'contains',word).df,
                      BM(datadic).select('Unnamed: 6', 'contains',word).df))

# Name of occupation for SOCP number
def socp_name(socc):
    return datadic[datadic['Record Type']== str(socc)]['Unnamed: 6'].values[0]


# The 6 coefficient/dimension correspond to the 6 occupation/feature clusters
# Functions for looking at the clusters:

tits = from_onet('Alternate Titles')
all_SOCP = set(tits['O*NET-SOC Code'])
aaa = tits[['O*NET-SOC Code','Title']].copy()
aaa['O*NET-SOC Code'] = aaa['O*NET-SOC Code'].apply(soc)
looktit = dict(aaa.values)

def lookup_title(socpnr):
    return looktit[socpnr]



#
class   Onet:
    def __init__(self,features):
        self.features = features
        self.data = from_onet(features)
        self.matrix = self.construct_matrix()
   
    def construct_matrix(self):
        '''
        Reshapes O*NET occupation/feature matrix to a regular matrix of N_occupations x M_features. Used as X for fitting. 
        '''
        prepdata = self.prepare()
        foo = pd.get_dummies(prepdata['Element Name']) 
        occ_prepdata = prepdata[['O*NET-SOC Code']].join(foo.multiply(prepdata['Data Value'],axis = "index")).groupby('O*NET-SOC Code').sum()
        occ_prepdata['SOCP'] = occ_prepdata.index
        occ_prepdata['SOCP_shave']=occ_prepdata['SOCP'].apply(soc)
        # Group by census/PUMS SOC codes (SOCP_shave)
        occ_prepdata_compounded= occ_prepdata.groupby('SOCP_shave').mean()
        occ_prepdata_compounded['SOCP_shave'] = occ_prepdata_compounded.index   
        foo = occ_prepdata_compounded.drop(columns='SOCP_shave')
        return foo
  
 # Prepare Matrix construction
    def prepare(self):
        '''
        Preprocesses O*Net data for shaping the features x occupations matrix (used as X)
        '''
        df = self.data
        # For Abilities, Knowledge, Skills
        if 'LV' in set(df['Scale ID']):
            sid = 'LV'
        # For Interests
        elif 'OI' in set(df['Scale ID']):
            sid = 'OI'
        df = df[df['Scale ID'] == sid]
        return df[['O*NET-SOC Code','Element Name','Data Value']] 

class   Census:

    def __init__(self,state = 'California'):
        self.state = state
        self.data = pd.read_pickle('data/pickle/pums_'+state+'.pkl')
        self.workers_occupations()


    def workers_occupations(self,age_low = 40, age_high = 65, std_max = 0.5, socp_granularity = 5):
        self.workers = self.data[
                            (self.data['AGEP'] >= age_low) &
                            (self.data['AGEP'] <= age_high)
                        ]
        self.workers['log FTE'] = self.workers['FTE wage'].apply(np.log)
        self.workers['SOCP_shave'] = self.workers['SOCP'].apply(lambda socpnr: soc(socpnr,socp_granularity))
        foo = self.workers.groupby('SOCP_shave') 
        self.occupations = foo.mean()[['AGEP', 'FTE wage','log FTE']]
        self.occupations['count'] = foo.count()['AGEP']
        self.occupations['std log FTE'] = foo.std()['log FTE']
        self.occupations['Occupation'] = self.occupations.index
        self.occupations_select = self.occupations[self.occupations['std log FTE'] < std_max]





## OBJECT-ORIENTED FITTING 
class Xfit:
    '''
    Xfit is a 'fit-as-an-object' solution:
        my_fit = Xfit(X,y, my_regressor, itr, xval) 
            does the following:
            1. SPLITS X and y into test and training sets.
            2. FITS a cross-validation, slicing the training data into 'xval' slices : cross_validate(regressor,X_train.values, y_train.values, cv=xval) 
            3. BOOTSTRAPS: Repeats (1-2) 'itr' number of times
            4. RETURNS RESULTS as attributes:
                my_fit.X          – List: The original X input data
                my_fit.itr        – Number of iterations / fits
                my_fit.y          – List: The original y input data
                my_fit.xval       – Number of slices in the cross validation
                my_fit.fit        – Dictionary: the 'itr' number of cross-validated fits, including estimators
                my_fit.y_test     – Dictionary: the y_test (list) for each fit
                my_fit.y_predict  – Dictionary: the predicted y for each fit
                my_fit.scores     – Pandas.DataFrame: validation scores for all fits 
                my_fit.score      – Dictionary: the mean score and standard deviation. 
                my_fit.features_importances        – Dictionary: feature_importances for all fits (for estimators with '.feature_importance_' as an attribute )
                my_fit.feature_importance          – Pandas.DataFrame: the average feature importances and standard deviations.           
    '''
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import cross_validate
    from sklearn.model_selection import cross_val_score
    from sklearn import linear_model
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    import xgboost as xgb  

    def __init__(self,X,y,regressor = xgb.XGBRegressor(),itr = 10, xval = 3):      
        # FITTING
        n = xval  
        feature_names = X.columns
        res = {}
        ypred = {}
        ytest = {}
        scor = {}
        feat_imp = {}       
        for i in range(itr):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            res_xboo = cross_validate(regressor,X_train.values, y_train.values, cv=n, return_estimator=True)
            ytest[i] = y_test
            res[i] = res_xboo
            ypred[i] = [res_xboo['estimator'][j].predict(X_test.values) for j in range(n)]
            scor[i] = [res_xboo['estimator'][j].score(X_test.values,y_test.values) for j in range(n)]
            feat_imp[i] = [res_xboo['estimator'][j].feature_importances_ for j in range(n)]
        scor_tot = np.concatenate(np.array(list(scor.values())))
        feat_tot = pd.concat([pd.DataFrame(feat_imp[i]) for i in range(itr)])
        feat_tot.columns = X.columns
        feat_tot.reset_index(inplace=True,drop = True)
        feat_mean = pd.concat([feat_tot.mean(),feat_tot.std()],axis=1)
        feat_mean.columns = ['mean','std']
        feat_mean['ratio'] = feat_mean['std']/feat_mean['mean']      
        # STORING RESULTS AS ATTRIBUTES
        self.X = X
        self.y = y
        self.fit = res
        self.y_predict = ypred
        self.y_test = ytest
        self.scores = pd.DataFrame(scor).T
        self.score = {'mean':scor_tot.mean(), 'std':scor_tot.std()}
        self.feature_importances = feat_imp
        self.feature_importance = feat_mean.sort_values('mean',ascending=False)
        self.itr =itr
        self.xv = xval


## MATRIX-FACTORIZATION: DIMENSIONALITY REDUCTION & ARCHETYPING

# CLUSTER FEATURES INTO OCCUPATION CATEGORIES
# Use non-zero matrix factorization for clustering
# Use singular value decomposition first state for determining overall similarity

class Archetypes:
    '''
    Archetypes: Performs NMF of order n on X and stores the result as attributes. 
    '''
    def __init__(self,X,n):
        self.n = n
        self.X = X
        self.model = NMF(n_components=n, init='random', random_state=0, max_iter = 1000, tol = 0.0000001)
        self.w = self.model.fit_transform(X)
        self.o = pd.DataFrame(self.w,index=X.index)
        self.on = nrmcol(self.o.T).T
        self.occ = self.on.copy()
        self.occ['Occupations'] = self.occ.index
        self.occ['Occupations'] = self.occ['Occupations'].apply(lookup_title)
        self.occ = self.occ.set_index('Occupations')
        self.h = self.model.components_
        self.f = pd.DataFrame(self.h,columns=X.columns)
        self.fn = nrmcol(self.f.T).T
        
    def plot_features(self): 
        sns.clustermap(self.fn.apply(lambda x: x**2).T,figsize=(self.n, self.X.shape[1]/3.5),method = 'single')

    def plot_occupations(self):
        sns.clustermap(self.occ.apply(lambda x: x**2),figsize=(self.n, self.X.shape[0]/3.5),method = 'single',z_score = 0)


class Svd:
    ''''
    Singular value decomposition-as-an-object
        my_svd = Svd(X) returns
        my_svd.u/.s/.vt – U S and VT from the Singular Value Decomposition (see manual)
        my_svd.f        – Pandas.DataFrame: features x svd_features
        my_svd.o        - Pandas.DataFrame: occupations x svd_features
    '''
    def __init__(self,X):
        self.u,self.s,self.vt = svd(np.array(X))
        self.f = pd.DataFrame(self.vt,columns=X.columns)
        self.o = pd.DataFrame(self.u,columns=X.index)
        



def plo(onet_matrix,n):
    '''
     preliminary 
    '''
    df = Archetypes(onet_matrix,n).on.apply(np.square)
    df['Title'] = df.index
    df['Title'] = df['Title'].apply(lookup_title)
    df = df.merge(occupations[['FTE wage','count']].astype(int),left_index = True, right_on='SOCP_shave')
    return df
    



#%%
if __name__ == "__main__":      
# # 
# OCCUPATIONAL ARCHETYPES 
# Purpose: A tool for people and organizations to discuss, analyze, predict, strategize on the job market. 
#
# Data Sources:
# - Occupation data: O*NET
# - People data: Census ACS/PUMS
#
# ## What this codes does
# - Creates occupational archetypes from a matrix of occupations and festures (abilities, skills, knowledge, etc.)
# - Selects a demography from census
# - Calculates labor market statistics, economic indicators for the archetypes with regards to the demography
# - Evaluate the predictive power of the Archtypes: 
#       - select a target (yearly full-time equivalent wage)
#       - fit features to target: compare Archetypes vs original features.
# 



    ## CENSUS DATA
    # select workers in ages 40 - 65 and discard the occupations with large standard deviations.

    census = Census('California')

    ## ONET DATA

    ab = Onet('Abilities')
    kn = Onet('Knowledge')
    sk = Onet('Skills')

    # Put them all together in one big feature matrix
    dall = pd.concat([ab.matrix,sk.matrix,kn.matrix],axis = 1)


    ## ESTIMATORS
    rid = linear_model.Ridge(alpha=.5)
    rf = RandomForestRegressor(n_estimators=40,
                            max_features='auto',
                            random_state=0)
    boo = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto', validation_fraction=0.1, n_iter_no_change=None, tol=0.0001)
    xboo = xgb.XGBRegressor()
