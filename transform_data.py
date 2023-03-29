# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 12:48:44 2023

@author: Jannik Sheikh
"""


import pandas as pd
import numpy as np

class DataProcessor:
    
    def __init__(self, df_list, names):
        self.df_list = df_list
        self.names = names
    
    def build_quantiles(self, df, name, chnky):

        df = pd.DataFrame(df)
        new_df = pd.DataFrame()
        
        low_p = np.nanquantile(df.iloc[:, 2:-1].astype('float64'), 0.25, axis = 1)
        mean = np.mean(df.iloc[:, 2:-1].astype('float64'), axis = 1)
        high_p = np.nanquantile(df.iloc[:, 2:-1].astype('float64'), 0.75, axis=1)
        
        new_df[f'{name}_25_quantile'] = low_p
        new_df[f'{name}_mean'] = mean
        new_df[f'{name}_75_quantile'] = high_p
        
        
        new_df[f'{name}_distance_to_total_mean'] = new_df[f'{name}_mean'] - (new_df[f'{name}_mean'].sum() / len(new_df))
        new_df[f'{name}_distance_to_75_quantile'] = new_df[f'{name}_75_quantile'] - (new_df[f'{name}_75_quantile'].sum() / len(new_df))
        new_df[f'{name}_distance_to_25_quantile'] = new_df[f'{name}_25_quantile'] - (new_df[f'{name}_25_quantile'].sum() / len(new_df))
        
        
        # new_df[f'{name}_len_data'] = df[df.notnull()].count(axis=1)

        return new_df

    def process_dataframes(self):
        
        

        # Create a list of numpy arrays from each dataframe
        arrays = [df.iloc[:, 3:-1].to_numpy() for df in self.df_list]

        # save all indcies of the chunky data
        index_list = self.df_list[0].index[self.df_list[0]['chunky'] == 1].tolist()

        result_df = []
        for i, arr in enumerate(arrays):
            result_df.append(self.build_quantiles(arr, self.names[i], index_list))
        
        final_df = pd.concat(result_df, axis = 1)


        final_df['Probenbezeichnung'] = self.df_list[0].Probenbezeichnung     
        final_df['proportion'] = self.df_list[0].proportion
        final_df['chunky'] = 0
        final_df.loc[(final_df['proportion'] > 0), 'chunky'] = 1
        final_df['ppm'] =  self.df_list[0].ppm        
  
        
        return final_df