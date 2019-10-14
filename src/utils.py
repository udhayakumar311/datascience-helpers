import pandas as pd
import numpy as np

from math import log10, floor

from io import BytesIO
import sys
import os
import re
from glob import glob
import pickle
import tempfile
import configparser

from IPython.display import HTML, display

import requests
import boto3
from PIL import Image

from functools import reduce

from collections import OrderedDict
from operator import itemgetter

import logging
import time
from datetime import datetime

def get_logger():
    # create logger
    logger = logging.getLogger('MAIN_LOGGER')
    logger.setLevel(logging.INFO)
    # create console handler and set level to debug
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    # create formatter
    formatter = logging.Formatter('%(asctime)s (%(levelname)s): %(message)s', 
                                  datefmt='%Y/%m/%d %I:%M:%S %p')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.handlers = [ch]
    return logger

############################
#  Simple Pickle Manager   #
############################

class PickleManager:
    '''Simply loads and saves objects using pickle'''

    @staticmethod
    def load(filepath):
        '''Load from a pickle object'''
        return pickle.load(open(filepath, 'rb'))
    
    @staticmethod
    def save(obj, filepath, overwrite=False):
        '''Saves (optionaly overwrites) to a given filepath.'''
        if overwrite is False:
            assert not os.path.isfile(filepath), "The filepath you are trying to write to already exists. Please choose another path or set overwrite to True"
        return pickle.dump(obj, open(filepath, 'wb'))

############################
#     S3 File Manager      #
############################

class S3Manager:
    '''This class abstracts away several ways of interacting with an S3 bucket. Namely
    
    - loading a file
    - saving a file
    - downloading a file
    - uploading a file
    - reading a csv
    - reading a pickle
    
    Note:
    =====
    This requires being on the Burberry network to access the bucket analytics.demandforecasting.data.
    '''
    
    def __init__(self, config_file, bucket, region_name):
        '''When a config file is not provided it will connect based on the enviroment variables'''
        
        # If a config file is provided, it should be parsable with configparser
        if config_file is not None:
            config = configparser.ConfigParser()
            config.read(config_file)

            aws_access_key_id = config['S3'].get('aws_access_key_id')
            aws_secret_access_key = config['S3'].get('aws_secret_access_key')

            self.region_name = region_name
        else:
            aws_access_key_id = None # Read from the enviroment variable
            aws_secret_access_key = None # Read from the enviroment variable
            
        self.bucket = bucket
        self.s3_resource = boto3.resource('s3', region_name=region_name,
                             aws_access_key_id=aws_access_key_id, 
                             aws_secret_access_key=aws_secret_access_key)
        
        
        
    def load(self, s3_filename):
        '''Load from file from s3 using pickle.loads().'''
        
        obj = self.read_pickle(s3_filename)
        
        return obj 
    
    def save(self, obj, s3_filename):
        '''Saves an object to s3 with the given filepath.'''
        
        temp_file = tempfile.NamedTemporaryFile()
        pickle.dump(obj, open(temp_file.name,'wb'))
        self.upload(temp_file.name, s3_filename)
        return self
    
    def download(self, s3_filename, local_filename=None):
        '''Downloads a file from s3. It will be saved with the same basename if no local_filename is provided.'''
        
        if local_filename is None:
            local_filename = os.path.basename(s3_filename)
        
        conn = self._get_conn(s3_filename)
        conn.download_file(local_filename)
        return self
    
    def upload(self, local_filename, s3_filename=None, use_image_gzip=False):
        '''Uploads a file to s3. It will be saved with the same basename if no target_filename is provided.'''
        
        if s3_filename is None:
            s3_filename = os.path.basename(local_filename)
            
        conn = self._get_conn(s3_filename)
            
        if use_image_gzip:
            conn.upload_file(Filename=local_filename, ExtraArgs={"ContentType": "image/svg+xml", "ContentEncoding": "gzip"})
        else:
            conn.upload_file(Filename=local_filename)

        return self
    
    def _get_conn(self, s3_filename):
        conn = self.s3_resource.Object(self.bucket, s3_filename)
        return conn
    
    def _get_body(self, s3_filename):
        '''Pings the s3 bucket and returns a bytestream to the body of the object.
        returns a botocore.response.StreamingBody object
        '''
        conn = self._get_conn(s3_filename)
        body = conn.get()['Body']
        return body
    
    def read_csv(self, s3_filename, **pd_kwargs):
        '''Uses pandas.read_csv to read the s3 file.'''
        body = self._get_body(s3_filename)
        return pd.read_csv(body, **pd_kwargs)
    
    def read_pickle(self, s3_filename):
        '''Same as self.load(), uses pickle.load() to read the s3 file.'''
        body = self._get_body(s3_filename)
        obj = pickle.loads(body.read())
        return obj

#####################################
#    Pretty display in notebook     #
#####################################

def display_multiple_from_dict(dict_of_pandas_dataframes, n_rows=5):
    '''Displays several dataframes at once.'''
    
    for key, dataframe in dict_of_pandas_dataframes.items():
        if n_rows:
            display(dataframe.head(n_rows))
        else:
            display(dataframe)
    
    return None

def plist(itterable, n_cols = 3, spacing=40):
    '''Print an itterable in n_columns.'''
    for i, val in enumerate(itterable):
        if i%n_cols == n_cols-1:
            print(str(val))
        else:
            print(str(val).ljust(spacing, ' '), end='')
            
def head(data, n_rows=5):
    '''Shows the first n rows of a spark.DataFrame.'''
    rows = data.take(n_rows)
    return pd.DataFrame(rows, columns=data.columns)

def code_toggle():
    '''HTML incerpt into notebook to display/hide all code cells.'''
    
    return HTML('''<script>code_show=true; function code_toggle() {
                   if (code_show){$('div.input').hide();} else {$('div.input').show();}
                   code_show = !code_show} $( document ).ready(code_toggle);</script>
                   To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')

#####################################
#   Dataframe column naming utils   #
#####################################

def drop_extra_columns(dataframe):
    '''Drops any columns that are all nan values and "Unnamed: x".'''

    columns = dataframe.columns
    
    unnamed_columns = [col for col in columns if col.startswith('Unnamed: ')]
    
    irrelevant_columns = []
    for col in unnamed_columns:
        if all(dataframe[col].isna()):
            irrelevant_columns.append(col)
            
    s = dataframe.drop(irrelevant_columns, axis=1)
    return s

def cast_col_names_to_underscore_style(dataframe, remove_newline=True, remove_punctuation=True):
    '''Replaces column names with underscore style naming.'''
    columns = dataframe.columns
    
    if remove_newline:
        columns = [str(col).replace('\n', ' ') for col in columns]
        
    if remove_punctuation:
        columns = [str(col).replace('.', '').replace(',','').replace(':','') for col in columns]

    
    columns = [re.sub(' +', '_', str(col)).lower().replace(' ', '_') for col in columns]
    
    s = dataframe.copy()
    s.columns = columns
    return s

##################################
#   Data Engineering Functions   #
##################################

def map_to_other(series, threshold=1, fill_with='OTHER'):
    '''If a value appears less times than the threshold it will be cast to the "other" category.
    
    If the threshold is an int, values that appear fewer times are replaced with "fill_with".
    If the threshold is a float, values that appear in fewer than '''
    
    if isinstance(threshold, int):
        x = series.value_counts()
    elif isinstance(threshold, float):
        x = series.value_counts(normalize=True)
    else:
        raise ValueError('Value for threshold must be a float or an int.')
        
    cast_to_other = set(x[lambda x: x <= threshold].index)
    
    series = series.apply(lambda x: fill_with if x in cast_to_other else x)
    return series

#################
#     MISC      #
#################

def safecall(f, default=None, exception=Exception):
    '''Returns modified f. When the modified f is called and throws an
    exception, the default value is returned'''
    def _safecall(*args,**kwargs):
        try:
            return f(*args,**kwargs)
        except exception:
            return default
    return _safecall

def get_time_now(asString=True):
    """
    Returns the time right now as year-month-day-hour-minute-second for example '2019-07-25-10-56-32' 
    
    Parameters:
    -----------
    - asString : True if you want to return the time in String Format
    """
    
    time_now_unix = time.time()
    date_now_datetime = datetime.fromtimestamp(time_now_unix)
    

    year   = str(date_now_datetime.year  ).rjust(2, '0')
    month  = str(date_now_datetime.month ).rjust(2, '0')
    day    = str(date_now_datetime.day   ).rjust(2, '0')
    hour   = str(date_now_datetime.hour  ).rjust(2, '0')
    minute = str(date_now_datetime.minute).rjust(2, '0')
    second = str(date_now_datetime.second).rjust(2, '0') 
    
    date_now_string = f'{year}-{month}-{day}-{hour}-{minute}-{second}'
    
    if asString:
        ans = date_now_string
    else:
        ans = date_now_datetime
        
    return ans

def get_most_recent_file(basename, directory='.', ending_in='', return_all=False, containing=r'\d{4}-\d{2}-\d{2}'):
    '''Searches a disrectory for a file matching f'{directory}/{basename}*{containing}*{ending_in}' 
    
    Returns:
    -------
    The path of the file with the latest (largest) string past the basename.
    '''
    
    start_of_path = os.path.join(directory, basename)
    
    files = glob(f'{start_of_path}*{containing}*{ending_in}')
    
    file_endings = [file.replace(start_of_path, '') for file in files]
    
    order = np.argsort(file_endings) # From smallest to larggest
    
    ordered_files = np.array(files)[order]
    
    if return_all:
        ans = ordered_files
    else:
        ans = ordered_files[-1] # the bigest should be the most recent
    
    return ans

def convert_bytes(num):
    """this function will convert bytes to MB.... GB... etc"""
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x) # Exits function here
        num /= 1024.0


def file_size(file_path, human_readable=False):
    '''Return the file size in bytes'''
    #if os.path.isfile(file_path):
    file_info = os.stat(file_path)
    byte_size = file_info.st_size
    
    if human_readable:
        ans = convert_bytes(byte_size)
    else:
        ans = byte_size
    return ans

def make_col2_nan_where_equal_to_col1(data, col1, col2=None, inplace=False, suffix='_alt', or_col1_starts_with_col2=True):
    '''This function is intended to be used to cast all _alt columns to NaN where they have the same value as the non-alt version.'''
    assert col1 != col2, 'Careful, if col1 == col2 col1 will be made all NaN.'
    
    if col2 is None:
        col2 = col1 + suffix
    
    if inplace is False:
        data = data.copy()
        
    data.loc[lambda x: (x[col1]==x[col2]) & ~(x[col1].isna()), col2] = np.nan
    
    if or_col1_starts_with_col2:
        if data[col1].dtype == 'object':
            data.loc[lambda x: (x[col1].str.startswith(x[col2])) & ~(x[col1].isna()), col2] = np.nan
    
    return data

def multimerge(*tables, on=None, how='outer'):
    '''Will recursively apply partial(pd.merge n=on, how=how) on the given tables.'''
    
    for table in tables:
        assert isinstance(table, pd.DataFrame), 'This function requires all tables to be of type pd.DataFrame.'
        
    merged_table = reduce(lambda table1, table2 : pd.merge(table1, table2, on=on, how=how), tables)
    
    return merged_table

def format_int_to_date(date_as_int):
    '''Takes an integer or string representing a date in format yyyymmdd and returns a string of the form yyyy-mm-dd'''
    
    date_as_number_string = (str(date_as_int))
    assert  len(date_as_number_string) == 8, f'The date needs to be in a yyyymmdd format. The imput was {len(date_as_number_string)} char long.'
  
    year = date_as_number_string[:4]
    month = date_as_number_string[4:6]
    day = date_as_number_string[6:]
    
    date = f'{year}-{month}-{day}'
    return date


def embed_df2_in_df1(df1, df2, priority=None):
    """

    This function:
    (1) Takes two dataframes, flattens them
    (2) Takes values in the same position in each data frame puts them next to each other in a string
    (3) Returns the strings in a list
 
    Parameters:
    -----------
    - df1 : A pandas dataframe
    - df2 : A pandas dataframe
    - priority: None if 
    
    """
    columns = df1.columns
    index = df1.index
    
    new_values = np.array([f'{i} ({j})' for i, j in zip(df1.values.ravel(), df2.values.ravel())])
    
    df = pd.DataFrame(new_values.reshape(df1.shape), index=index, columns=columns)
    
    if not priority is None:
        df = df.loc[:, ['OVERALL']+list(priority)]
        
    return df


def round_sig(x, sig=2):
    """Returns a value rounded to a specified number of significant figures
    
    Parameters:
    -----------
    - x : the value you want to round
    - sig : the number of significant figures you want to round x to
    
    """
    return round(x, sig-int(floor(log10(abs(x))))-1)


def get_example_column_values(df):
    '''Returns a table of the form with the most common 5 values in each column and their % occurence]
    
    Example:
    --------
    
    >>> df = DataFrame({'A':[1,1,2,1,2], 'B':[2,2,3,4,2]})
    >>> get_example_column_values(df)
    
    |    A    |    B    |
    |---------|---------|
    |[1, .60] |[2, .60] |
    |[2, .40] |[3, .20] |
    |         |[4, .20] |
    '''
    
    examples = {}
    for col in df.columns:
        
        uniques = ['', '', '', '', '']
        for i, val in enumerate(df[col].value_counts(dropna=False, normalize=True).round(2).head(5).reset_index().values):
            uniques[i]=val
        
        
        examples[col]=uniques
        
    return pd.DataFrame(examples)

def make_non_negative(x):
    '''return max(x, 0)'''
    return max(x, 0)

def prettify_dataframe(df, styling_name='dark_header'):
    '''Displays a given dataframe in 
    if styling_name == 'dark_header':
        df = df.style.set_table_styles([
            {'selector': 'tr:nth-of-type(odd)','props': [('background', '#eee')]}, 
            {'selector': 'tr:nth-of-type(even)', 'props': [('background', 'white')]},
            {'selector': 'th', 'props': [('background', '#606060'), ('color', 'white'),('font-family', 'verdana')]},
            {'selector': 'td', 'props': [('font-family', 'verdana')]},
                ])
        
    else:
        raise ValueError(f'Style {styling_name} not implemented, please choose one of []"dark_header"]')
        
    return df

def class_to_dict(obj, dtype=pd.DataFrame):
    '''Takes an onject "obj" and creates a dictionary of the attributes of obj of type dtype.
    
    Example:
    --------
    >>> helper = ProjectData(sqlContext).run()
    >>> x = class_to_dict(helper)
    >>> x
    {
        'df_tdmd': ___,
        'df_sales': ___,
        ...
    }
    '''
    ans = {}
    for key in obj.__dict__.keys():
        if isinstance(obj.__getattribute__(key), dtype):
            ans[key]=obj.__getattribute__(key)
    return ans

