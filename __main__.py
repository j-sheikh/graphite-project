# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 10:05:59 2023

@author: jannik sheikh
"""

import sys
sys.path.append('Excel to CSV')
sys.path.append('Transform Data to Quantile')

import os
from read_graphite_data import get_data
from transform_data import DataProcessor
from tqdm import tqdm
import pandas as pd


class Preprocessing:
    def __init__(self, main_folder):
        self.main_folder = main_folder


    def process_original_data(self, item):

        file_name = 'Urwerte_V1_V9_neu.xlsx'
        full_path = os.path.join(item, file_name)
        
        # Get the data for each sheet in the Excel file
        df0 = get_data(full_path, 'Feret-Verhältnis')
        df1 = get_data(full_path, 'Graphitfläche')
        df2 = get_data(full_path, 'Kompaktheit')
        df3 = get_data(full_path, 'Konvexität')
        df4 = get_data(full_path, 'NN-Abstand')
        df5 = get_data(full_path, 'Rundheit')
        df6 = get_data(full_path, 'Sphärizität')
        
    
        # Save the data to CSV files in the same directory as the Excel file
        df0.to_csv(os.path.join(item, 'feret_data.csv'), index=False)
        df1.to_csv(os.path.join(item, 'graphite_area_data.csv'), index=False)
        df2.to_csv(os.path.join(item, 'compact_data.csv'), index=False)
        df3.to_csv(os.path.join(item, 'convexity_data.csv'), index=False)
        df4.to_csv(os.path.join(item, 'nn_distance_data.csv'), index=False)
        df5.to_csv(os.path.join(item, 'roundness_data.csv'), index=False)
        df6.to_csv(os.path.join(item, 'sphericity_data.csv'), index=False)
        
        
        dataframes = [df0, df1, df2, df3, df4, df5, df6]
        names = ['feret', 'graphite', 'compact', 'convexity', 'nn', 'roundness', 'sphericity']
        df = DataProcessor(dataframes, names).process_dataframes()
        df.to_csv(os.path.join(item, 'df_quantiles.csv'), index=False)
        
    
    def process_subfolder(self, subfolder):

        csv_files = [f for f in os.listdir(subfolder) if f.endswith(".csv") and f not in ['df_quantiles.csv']]
    
        if len(csv_files) == 0:
            return
        dataframes = [pd.read_csv(os.path.join(subfolder, csv_file)) for csv_file in csv_files]
        names = [filename.split('_')[1] for filename in csv_files]    
        df = DataProcessor(dataframes, names).process_dataframes()
        
    
        df.to_csv(os.path.join(subfolder, 'df_quantiles.csv'), index=False)
    
    def process_folder(self, folder):
        total_files = self.count_files(folder)
        for item in tqdm(os.scandir(folder), desc="Processing Folder", total=total_files):
            
            if item.is_dir() and item.name in ['original_dataset']:
                self.process_original_data(item)
            elif item.name.endswith(".csv"):
                self.process_subfolder(folder)
                break   
            elif item.is_dir() and item.name not in ['original_dataset']:
                self.process_folder(item.path)
                
    def count_files(self, path):
        count = 0
        for root, dirs, files in os.walk(path):
            count += len(files)
        return count
    
    def run(self):
        self.process_folder(self.main_folder)
    

path = r'XXX'

ca = Preprocessing(path)
ca.run()

