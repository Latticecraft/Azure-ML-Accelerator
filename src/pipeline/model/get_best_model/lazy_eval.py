import pandas as pd


class LazyEval:
    def __init__(self, dict_files):
        self.dict_files = dict_files

    def get(self, fold, imputer, balancer='none'):
        df_X = self.dict_files[f'X_{fold}']
        df_y = self.dict_files[f'y_{fold}']
        
        cols = df_X.columns

        df_X = self.dict_files[f'imputer____{imputer}'].fit_transform(df_X)
        if balancer != 'none':
            df_X, df_y = self.dict_files[f'balancer____{imputer}_{balancer}'].fit_resample(df_X, df_y)

        df_X = pd.DataFrame(df_X, columns=cols)
        
        return df_X, df_y