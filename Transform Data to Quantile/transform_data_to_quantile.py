# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 16:36:36 2022

@author: hikth
"""

import pandas as pd
import numpy as np

df0 = pd.read_csv('compact_data.csv')
df1 = pd.read_csv('convexity_data.csv')
df2 = pd.read_csv('feret_data.csv')
df3 = pd.read_csv('graphite_area_data.csv')
df4 = pd.read_csv('nn_distance_data.csv')
df5 = pd.read_csv('roundness_data.csv')
df6 = pd.read_csv('sphericity_data.csv')

# original = pd.read_csv('quantile_data_extended.csv')

df0.loc[0, 'chunky'] = 'N'
df0_array = np.array(df0.iloc[:, 2:-1])
chnky = df0.loc[df0.chunky == 'Y'].index.to_list()


df1.loc[0, 'chunky'] = 'N'
df1_array = np.array(df1.iloc[:, 2:-1])



df2.loc[0, 'chunky'] = 'N'
df2_array = np.array(df2.iloc[:, 2:-1])


df3.loc[0, 'chunky'] = 'N'
df3_array = np.array(df3.iloc[:, 2:-1])


df4.loc[0, 'chunky'] = 'N'
df4_array = np.array(df4.iloc[:, 2:-1])


df5.loc[0, 'chunky'] = 'N'
df5_array = np.array(df5.iloc[:, 2:-1])


df6.loc[0, 'chunky'] = 'N'
df6_array = np.array(df6.iloc[:, 2:-1])




def build_quantiles(df, name, chnky=[]):
    df = pd.DataFrame(df)
    new_df = pd.DataFrame()
    for i in range(len(df)):
       
        low_p = df.iloc[i, :].quantile(0.25)
        mean = df.iloc[i, :].mean()
        high_p = df.iloc[i, :].quantile(0.75)
           
        # print(name)
        new_df.loc[i, f'{name}_25_quantile'] = low_p
        new_df.loc[i, f'{name}_mean'] = mean
        new_df.loc[i, f'{name}_75_quantile'] = high_p
        
    new_df[f'{name}_distance_to_total_mean'] = new_df[f'{name}_mean'] - (new_df.drop(chnky)[f'{name}_mean'].sum() / len(new_df.drop(chnky)))
    new_df[f'{name}_distance_to_75_quantile'] = new_df[f'{name}_75_quantile'] - (new_df.drop(chnky)[f'{name}_75_quantile'].sum() / len(new_df.drop(chnky)))
    new_df[f'{name}_distance_to_25_quantile'] = new_df[f'{name}_25_quantile'] - (new_df.drop(chnky)[f'{name}_25_quantile'].sum() / len(new_df.drop(chnky)))
    new_df[f'{name}_len_data'] = df[df.notnull()].count(axis=1)
        
        
    return new_df


df0_new = build_quantiles(df0_array, name = 'compact')

df1_new = build_quantiles(df1_array, name = 'convexity')

df2_new = build_quantiles(df2_array, name = 'feret')

df3_new = build_quantiles(df3_array, name = 'area')

df4_new = build_quantiles(df4_array,name = 'nn_distance')

df5_new = build_quantiles(df5_array, name = 'roundness')

df6_new = build_quantiles(df6_array, name = 'sphericity')



final_df = pd.concat([df0_new, df1_new, df2_new, df3_new, df4_new, df5_new, df6_new],axis = 1)

prop = {'V9_6': 0.7, 'V9_5':0.01, 'V4_2':0.3, 'V9_4':0.075, 'V9_3':0.185, 'V1_2':0.6, 'V9_2': 0.365, 'V9_1':0.807}




final_df['ppm'] = df0.ppm
final_df['chunky'] = df0.chunky.map(dict(Y = 1, N = 0))
dropcol = ['compact_len_data', 'convexity_len_data', 'feret_len_data', 'area_len_data', 'nn_distance_len_data', 'sphericity_len_data']
final_df.drop(columns = dropcol, inplace = True)
final_df = final_df.rename(columns={'roundness_len_data':'len_data'})
final_df.columns
final_df['Probenbezeichnung'] = df0.Probenbezeichnung
final_df['proportion'] = final_df.Probenbezeichnung.map(prop).fillna(0)


final_df.to_csv('new_data_quantiles.csv', index = False)
