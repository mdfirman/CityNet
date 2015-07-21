import inspect
import hashlib
import os
import pandas as pd
import numpy as np

def merge_two_dicts(x, y):
    '''Given two dicts, merge them into a new dict as a shallow copy.
    from http://stackoverflow.com/a/26853961/2156909'''
    z = x.copy()
    z.update(y)
    return z

class AzurePipeline(object):
    """ Class to mimic Azure behaviour """
    
    def __init__(self, meta, cache_dir='.', save_intermediate_results=True):
        """
        meta (dict(dict)):
                {function-name1: {arg1: value1,
                                  arg2: value2},
                 function-name2: {arg1: value1}}
                 
        cache_dir (string):
                folder for caching data
                
        save_intermediate_results (bool):
                save (cache) intermediate results [yes/no]
        """
        
        self.meta = pd.DataFrame(meta)
        self.df = None
        
        self.func_hash_dir = os.path.join(os.path.realpath(cache_dir),
                                          'func_hash')
        self.df_hash_dir = os.path.join(os.path.realpath(cache_dir),
                                        'df_hash')
        self.df_cache_dir = os.path.join(os.path.realpath(cache_dir),
                                         'df_cache')
        
        self.save_intermediate_results = save_intermediate_results        
        self.func_history = []
        
    
    def __add__(self, other):  
        """ overloaded + operator. Returns a merged AzurePipeline object"""
        d = merge_two_dicts(self.meta.to_dict(), other.meta.to_dict())
        
        ap = AzurePipeline(d)
        
        if other.df is not None:
            if self.df is None:
                ap.df = other.df
            else:
                ap.df = pd.concat((self.df, other.df), copy=True)
        
        ap.func_history = [self.func_history,
                             other.func_history]
        
        return ap
        
        
    def get_df_hash_file(self, func):
        """ get filename of DataFrame hash """
        try:
            args = self.meta[func.__name__].dropna().to_dict()
        except KeyError:
            args = ''
            
        filename = os.path.join(self.df_hash_dir, 
                            hashlib.sha256('-'.join(self.func_history + [func.__name__]) 
                                            + '-' + str(args)).hexdigest()
                            + '.sha256')
        
        if not os.path.exists(self.df_hash_dir):
            os.makedirs(self.df_hash_dir)
            
        return filename
        
    
    def get_func_hash_file(self, func):
        """ get filename of function source hash """
        filename = os.path.join(self.func_hash_dir, func.__name__ + '.sha256')
    
        if not os.path.exists(self.func_hash_dir):
            os.makedirs(self.func_hash_dir)
            
        return filename
        
        
    def get_df_cache_file(self, func):
        """ get filename of DataFrame cache (data) """
        try:
            args = self.meta[func.__name__].dropna().to_dict()
        except KeyError:
            args = ''
            
        filename = os.path.join(self.df_cache_dir, 
                            hashlib.sha256('-'.join(self.func_history + [func.__name__]) 
                                            + '-' + str(args)).hexdigest()
                            + '.pkl')   
    
        if not os.path.exists(self.df_cache_dir):
            os.makedirs(self.df_cache_dir)
            
        return filename
    
    
    def is_data_unchanged(self, func):
        """ check whether the data has already been used as argument of func """
        local_hash = hashlib.sha256(np.asarray(self.df).tostring()).hexdigest()
        
        filename = self.get_df_hash_file(func)
        
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                saved_hash = f.readline()
        else:
            saved_hash = None
            
        if local_hash == saved_hash:
            return True
        else:
            if self.save_intermediate_results:
                with open(filename, 'w') as f:
                    f.write(local_hash)
            
            return False           
    
    def is_func_unchanged(self, func):
        """ check whether the source code of func has remained stable """
        source = inspect.getsource(func)
        local_hash = hashlib.sha256(source).hexdigest()
        
        filename = self.get_func_hash_file(func)
        
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                saved_hash = f.readline()
        else:
            saved_hash = None
            
        if local_hash == saved_hash:
            return True
        else:
            if self.save_intermediate_results:
                with open(filename, 'w') as f:
                    f.write(local_hash)
            
            return False           
        
            
    def cache(self, func):
        """ dump DataFrame out to cache """
        filename = self.get_df_cache_file(func)
        self.df.to_pickle(filename)
    
    
    def load_from_cache(self, func):
        """ load previous computation from cache """
        filename = self.get_df_cache_file(func)
        self.df = pd.read_pickle(filename)
        
        
    def apply(self, func, args=None):
        """ apply func to pipeline, using complementary args if necessary """
        if args:
            d = merge_two_dicts(self.meta.to_dict(), args)
            self.meta = pd.DataFrame(d)

        # evaluate outside of condition to make sure
        # side effects are taking place
        data_unchanged = self.is_data_unchanged(func)
        func_unchanged = self.is_func_unchanged(func)
        
        if not data_unchanged or not func_unchanged:
            self.df, self.meta = func(self.df, self.meta,)        
            self.cache(func)
        else:
            self.load_from_cache(func)
            
        self.func_history += [func.__name__]
        