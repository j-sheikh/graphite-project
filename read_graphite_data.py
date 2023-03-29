# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 15:58:02 2022

@author: Jannik Sheikh
"""

import pandas as pd

def get_data(xls, name):
    
    df = pd.read_excel(xls, name, skiprows= range(1, 4), usecols="B:BD", header= 1)
    label = pd.read_excel(xls, name, usecols="B:BD", header= 0, skiprows= range(1, 3))
    labels = label.iloc[0, :].to_list()
       
    df = df.T.reset_index().rename(columns={'index':'Probenbezeichnung', 0:'ppm'})
    df.loc[0, 'Probenbezeichnung'] = 'V_perfect'
    df.loc[0, 'ppm'] = '0ppm'
    df['chunky'] = labels
    df.loc[0, 'chunky'] = 'N'   
    
    prop = {'V9_6': 0.7, 'V9_5':0.01, 'V4_2':0.3, 'V9_4':0.075, 'V9_3':0.185, 'V1_2':0.6, 'V9_2': 0.365, 'V9_1':0.807}
    
    df['proportion'] = df.Probenbezeichnung.map(prop).fillna(0)

    df['ppm'] = df.ppm.str.replace('ppm','')
    df['ppm'] = df.ppm.str.replace('pmm','')
    df['ppm'] = df.ppm.astype(int)

    
    
    
    return df

