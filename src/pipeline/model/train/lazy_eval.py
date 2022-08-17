class LazyEval:
    def __init__(self, dict_files):
        self.dict_files = dict_files

    def get(self, fold, imputer, balancer='none'):
        if balancer == 'rus':
            df_X = self.dict_files[f'X_{fold}_rus']
            df_y = self.dict_files[f'y_{fold}_rus']
        else:
            df_X = self.dict_files[f'X_{fold}_none']
            df_y = self.dict_files[f'y_{fold}_none']
            
            df_X = self.dict_files[f'imputer____{imputer}_none'].fit_transform(df_X)
            if balancer != 'none':
                df_X, df_y = self.dict_files[f'balancer____{imputer}_{balancer}'].fit_resample(df_X, df_y)
        
        return df_X, df_y