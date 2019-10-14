from functools import reduce
import os

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

from catboost import CatBoostRegressor, Pool
import shap

from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
import dask.bag as bag

from IPython.display import display, HTML
from PIL import Image

from src.utils import get_logger, PickleManager, get_time_now

#from src.evaluation import eval_predictions, eval_errors, eval_change, eval_box_plot
#from src.image_utils import display_dataframe

catboost_config = dict(n_estimators=1000,
                   allow_writing_files=True,
                   save_snapshot=True,
                   snapshot_file='experiment.snapshot',
                   snapshot_interval=200,
                   train_dir='logs',
                   metric_period=100,
                   nan_mode='Max',
                   learning_rate=0.1,
                   use_best_model=True,
                   )


def get_cat_feature_indices(X):
    '''Returns the index for each object column in a pandas dataframe'''
    return [i for i, is_object in enumerate(X.dtypes == 'object') if is_object]

class GroupByModel(BaseEstimator):
    '''This "model" uses a list of explicitly stated columns from the training data
    to do a groupby and then predicts the aggregated value from the appropiate group. 
    
    During training, for each group in the groupby the "target" column will be aggregated using "agg_func".
    The aggregated value for each group is remembered.
    
    During inference, the values in the "columns" is used to find what group the sample belongs to, and the 
    aggregated target value for that group is returned as the prediction.
    '''
    
    def __init__(self, columns:list, agg_func:str=np.mean):
        self.agg_func = agg_func
        self.columns = columns
        
    def fit(self, X, y):
        X = X.copy()
        assert '__target__' not in self.columns, "I don't think you should be predicting y using y as a predictor."
        
        X['__target__'] = y
        
        self.memory = pd.concat([X,y], axis=1).groupby(self.columns)['__target__'].apply(self.agg_func)
        return self
    
    def predict(self, X):
        y_pred = X[self.columns].merge(self.memory.reset_index(), on=self.columns, how='left')['__target__']
        return y_pred        
        
    def score(self, X, y, metric=r2_score):
        y_pred = self.predict(X)
        score = metric(y, y_pred)
        return score
    
class MultiGroupByModel(BaseEstimator):
    '''Extends the GroupByModel. 
    
    This implementation instanciates one version of the GroupByModel with each of the columns in "columns".
    It will then use the right-most column (assumed to be the most detailed?) to predict, but if this turns out
    to be NaN it will use the second-to-last column, etc...
    
    Example:
    ========
    
    y_pred = MultiGroupByModel(['level2_desc', 'level3_desc', 'level4_desc']).fit(X, y).predict(X)
 
    
    # IS EQUIVALENT TO 
    
    y_pred_level2 = GroupByModel(['level2_desc']).fit(X,y).predict(X)
    y_pred_level3 = GroupByModel(['level3_desc']).fit(X,y).predict(X)
    y_pred_level4 = GroupByModel(['level4_desc']).fit(X,y).predict(X)
    
    y_pred = y_pred_level4.fillna(y_pred_level3).fillna(y_pred_level2)
    
    # AND
    
    y_pred = MultiGroupByModel(['level2_desc', 'level3_desc', 'level4_desc']).fit(X, y).predict(X, y)

    # IS EQUIVALENT TO
    
    y_pred = y.fillna(y_pred_level4).fillna(y_pred_level3).fillna(y_pred_level2)

    '''
    
    def __init__(self, columns, agg_func=np.mean):
        self.columns = columns
        self.agg_func = agg_func
        self.models = [GroupByModel(columns=[col], agg_func=agg_func) for col in reversed(columns)]
        
    def fit(self, X, y):
        '''Calls .fit() in each of the internal GroupByModel instances.'''
        for model in self.models:
            model.fit(X,y)
            
        return self
    
    def predict(self, X, y=None):
        """If y is provided, y values will be returned direcly from y where y is not np.nan."""
        y_pred_s = [model.predict(X) for model in self.models]
        
        if not y is None:
            y_pred_s = [y] + y_pred_s
            
        y_pred = reduce(lambda x1, x2: x1.fillna(x2), y_pred_s)
        return y_pred
    
class LogModelWraper(BaseEstimator):
    '''Applies Log Scaling to the target variable, and undoes the scaling at inference time.'''
    def __init__(self, model):
        self.model = model
        
    def __repl__(self):
        return self.__repl__()+'_LOG'
        
    def fit(self, X, y, **kwargs):
        y_log = np.log(y+1)
        self.model.fit(X, y_log, **kwargs)
        return self
        
    def predict(self, X):
        y_log_pred = self.model.predict(X)
        y_pred = np.exp(y_log_pred)-1
        return y_pred
    
    def get_model_attribute(self, attribute):
        return self.model.__getattribute__(attribute)

    def get_model_method(self, method, *args, **kwargs):
        return self.model.__getattribute__(method)(*args, **kwargs)
    
class SqrtModelWraper(BaseEstimator):
    '''Applies Sqrt Scaling to the target variable, and undoes the scaling at inference time.'''
    def __init__(self, model):
        self.model = model
        
    def fit(self, X, y, **kwargs):
        y_sqrt = np.sqrt(y)
        self.model.fit(X, y_sqrt, **kwargs)
        return self
        
    def predict(self, X):
        y_sqrt_pred = self.model.predict(X)
        y_pred = y_sqrt_pred**2
        return y_pred
    
    def get_model_attribute(self, attribute):
        return self.model.__getattribute__(attribute)

    def get_model_method(self, method, *args, **kwargs):
        return self.model.__getattribute__(method)(*args, **kwargs)

class Experiment:
    '''Consolidates X, y, X_train, y_train, X_test, y_test, model and y_pred'''

    def __init__(self, model, X, y, 
                 data = None,  
                 bookkeeping_data = None,
                 train_set = None, 
                 test_set = None, 
                 eval_set = None,
                 name='experiment',
                ):

        self.model = model
        self.X = X
        self.y = y
        self.data = data
        self.bookkeeping_data = bookkeeping_data
        self.name = name
        
        self.train_set = train_set
        self.test_set = test_set
        self.eval_set = eval_set
        
        self.build()
        
    def build(self):
        '''This calls several private class methods to build the internal data tables.'''
        self._build_full_and_bookeeping_data()
        self._build_train_test_eval_sets()
        self._build_training_data()
        self._build_test_data()
        self._build_eval_data()
        self._validate_init()
        
    def run(self, *fit_args, **fit_kwargs):
        self.model.fit(self.X_train, self.y_train, *fit_args, **fit_kwargs)
        self.y_pred = self.predict(self.X_test)
        return self
    
    def fit(self):
        self.model.fit(self.X_train, self.y_train)
        return self
    
    def predict(self):
        '''Executes pd.Series(self.model.predict(self.X_test).ravel(), index=self.X_test.index) and saves this to y_pred'''
        y_pred = self.model.predict(self.X_test)
        self.y_pred = pd.Series(y_pred.ravel(), index=self.X_test.index)
        return self.y_pred
    
    def predict_proba(self):
        y_pred_proba = self.model.predict_proba(self.X_test)
        self.y_pred_proba = pd.DataFrame(y_pred_proba, index=self.X_test.index, columns=self.model.classes_)
        return self.y_pred
    
    def get_params(self):
        return self.model.get_params()
    
    def _build_full_and_bookeeping_data(self):
        if self.bookkeeping_data is None:
            self.bookkeeping_data = self.data
        if self.data is None:
            self.data = self.bookkeeping_data

    def _build_train_test_eval_sets(self):
        
        if self.test_set is None:
            self.test_set = ~self.train_set
            
        if self.eval_set is None:
            self.eval_set = self.test_set

    def _build_training_data(self):
        if isinstance(self.train_set, pd.Series):
            self.X_train = self.X.loc[self.train_set]
            self.y_train = self.y.loc[self.train_set]
            
    def _build_test_data(self):
        if isinstance(self.test_set, pd.Series):
            self.X_test = self.X.loc[self.test_set]
            self.y_test = self.y.loc[self.test_set]
            
    def _build_eval_data(self):
        if isinstance(self.eval_set, pd.Series):
            self.X_eval = self.X.loc[self.eval_set]
            self.y_eval = self.y.loc[self.eval_set]
            
    def _validate_init(self):
        
        assert isinstance(self.data, pd.DataFrame)
        assert isinstance(self.bookkeeping_data, pd.DataFrame)
        assert isinstance(self.X, pd.DataFrame)
        assert isinstance(self.y, pd.Series)
        
        assert all(self.data.index == self.bookkeeping_data.index), 'There should be a 1-to-1 matching between the records in data and in bookeeping_data, as given by row index in the pandas dataframe'
        assert all(self.X.index == self.y.index)
        
        assert set(self.X.index) <= set(self.train_set.index)
        assert set(self.X.index) <= set(self.test_set.index)
        assert set(self.X.index) <= set(self.eval_set.index)
        
        assert self.train_set.dtype == bool
        assert self.test_set.dtype == bool
        assert self.eval_set.dtype == bool
        
        assert set(self.X.index) <= set(self.data.index)
        assert set(self.bookkeeping_data.index) <= set(self.data.index)

    def save(self, filename=None, overwrite=False):
        '''Saves this object as a pickle file named.
        
        Parameters:
        -----------

        filename: str (default None)
            Name under which to save the object (with a .p extension).
            If None then filename defaults to self.name
        
        overwrite: bool (default False)
            If False, this method will raise and error of {filename}.p already exists.
            Set it equal to True to overwrite the file.
        '''

        if filename is None:
            filename = self.name+'.p'
            
        PickleManager.save(self, filename, overwrite=overwrite)
        return self

    def set_notes(self, text, mode='append', author=''):
        '''Adds notes to the notes field.
        
        Parameters:
        -----------
        text: str
            Text to add to the notes
            
        mode: str (default 'append')
            One of ['append','overwrite'].
            If mode=='append' the text is appended to the existing notes
            If mode=='overwrite' the text replaces the existing notes

        author: str (default '')
            Used to sign the note when provided
        ''' 
        note = f'''Time: {get_time_now()}\nBy: {author}{'-'*20}\n{text}\n\n'''

        if mode=='append':
            self.notes += note
        elif mode=='overwrite':
            self.notes = note
        else:
            raise ValueError(f'mode must be one of ["append", "overwrite"], not {mode}.')
            
        return self

class ShapExplainer(Experiment):
    '''The ShapExplainer class must derive from Experiment to make use of validate init and 
    the .build() method'''
   
    def set_shap_values(self):
        '''NOTE: This only works if self.model is an instance of CatBoostRegressor
        or CatBoostClassifier. It is possible to extend this to other models, but 
        only Catboost is currently supported.'''

        data_pool=Pool(self.X, cat_features=self.model.get_cat_feature_indices())
        
        full_shap_values = (
            self.model
            .get_feature_importance(data_pool, prettified=True, type='ShapValues')
            .rename(columns={i:col for i,col in enumerate(list(self.X.columns) + ['shap_expected_values'])})
            .set_index(self.X.index)
                           )

        
        self.shap_values = full_shap_values.iloc[:,:-1]
        self.shap_expected_values = full_shap_values.iloc[:,-1]
        
        self.set_shap_values_units()
        
        return self
        
    def explain_shap(self, record_index=None, **kwargs):
        '''Displays the Shap ForcePlot for a given index. 
        If no record index is provided, one is chosen at random.
        
        Parameters:
        -----------
        record_index: 
            Index of the underlying self.X table.

        Note:
        -----
        If no self.shape_values is None, this function will call self.set_shao_values()
        '''
        
        if self.shap_values is None:
            self.set_shap_values()
                     
        sample_shap_values = self.shap_values.loc[record_index,:].values
        expected_value = self.shap_expected_values.loc[record_index]
        features = self.X.loc[record_index,:]

        display(shap.force_plot(expected_value, sample_shap_values, features, **kwargs))
        return self

class ExtendedCatBoost(Experiment):
    ''' '''

    def build_model_affinity_table(self, verbose=0, logger=get_logger(), n_jobs=16):
        '''Uses the build_model_affinity_table() function to build a model-driven
        affinity between samples in X_train and samples in X_test'''
        values = build_model_affinity_table(model=self.model, 
                                            X_train = self.X_train, 
                                            X_test = self.X_test, 
                                            verbose=verbose, 
                                            logger=logger, n_jobs=n_jobs)

        dataframe = pd.DataFrame(data=values, 
                                 index=self.X_test.index, 
                                 columns=self.X_train.index)
        
        self.model_affinity_table = dataframe 

        return self


########################################
#    Other Model Relevant Functions    #
########################################

def build_model_affinity_table(model, X_train, X_test, n_jobs=16, verbose=0, weights=None, logger=get_logger()):
    '''Calculates the affinity between each sample in X_train and each sample in X_test by lookingt the fequency 
    in which the two samples end up in the same eastimator leaf node across the estimators in the (CatBoost) model.
    '''
    
    if verbose > 0: logger.info('Calculating leaf nodes over X_train')
    x_train_leafs = np.stack(model.iterate_leaf_indexes(X_train))

    if verbose > 0: logger.info('Calculating leaf nodes over X_test')
    x_test_leafs = np.stack(model.iterate_leaf_indexes(X_test))

    if verbose > 0: logger.info('Instanciating dask bag')
    x_test_leaf_bag = bag.from_sequence(x_test_leafs, npartitions=n_jobs)

    if weights is None:
        if verbose > 0: logger.info('Calculating average affinity beteen each x_train and x_test in X_train and X_test')
        affinity_list_of_lists = x_test_leaf_bag.map(lambda i: (i == x_train_leafs).mean(axis=1)).compute()
    else:
        assert len(weights)==model.tree_count_, 'When using weights please pass a weight for each tree.'
        if verbose > 0: logger.info('Calculating weighted affinity beteen each x_train and x_test in X_train and X_test')
        affinity_list_of_lists = x_test_leaf_bag.map(lambda i: np.multiply(i == x_train_leafs, weights).mean(axis=1)).compute()
    
    if verbose > 0: logger.info('Stacking into np.array of shape len(X_test) by len(X_train)')
    affinity_table = np.stack(affinity_list_of_lists)
    
    return affinity_table