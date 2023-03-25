# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 12:44:13 2023

@author: jannik sheikh
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans

from scipy.spatial.distance import pdist, squareform

import os
from itertools import product
import pickle



import warnings

# Filter out the DeprecationWarnings
warnings.filterwarnings("ignore", category=UserWarning, message="KMeans is known to have a memory leak on Windows with MKL")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
    

os.environ['OMP_NUM_THREADS'] = '1'


def run_hyperparamter_grid_model(df):
    param_grid = {
        'KMeans': {
            'n_clusters': [10 ,11, 12 ,13 ,14 ,15],
            'init': ['k-means++'],
            'n_init': [1, 10, 20, 30],
            'max_iter': [50, 100, 200, 300],
            'tol': [1e-1, 1e-2, 1e-3, 1e-4],
            'algorithm': ['auto'],
            'random_state':[0, 1, 36, 123]
        }
    }

    X = StandardScaler().fit_transform(df.iloc[:, :-4])
    # Define the models to evaluate
    models = {
        'KMeans': KMeans
    }
    # Define a dictionary to store the best scores and parameters for each model
    scores = {}


    # Loop over each model and its corresponding parameter grid
    for model_name, model_class in tqdm(models.items()):
        params = param_grid[model_name]
        param_names = list(params.keys())

        # Create a list of all possible combinations of parameter values
        param_values = list(product(*list(params.values())))


        eval_params = {}

        # Loop over each combination of parameter values
        for param_set in tqdm(param_values):
            # Create a dictionary of the current parameter values
            param_dict = {param_names[i]: param_set[i] for i in range(len(param_names))}

            # Create an instance of the current model with the current parameter values
            model = model_class(**param_dict)

            # Fit the model
            model.fit(X)

            labels = model.labels_
            
            prediction_df = pd.DataFrame(labels, columns=['cluster'])
            
        
            
            eval_params[f"{model_name}_{param_dict['n_clusters']}_{param_dict['n_init']}_{param_dict['max_iter']}_{param_dict['tol']}_{param_dict['random_state']}"] = (prediction_df, param_dict)

        # Store the best score and parameters for this model in the dictionary
        scores[model_name] = (eval_params)
        
    return scores
        
def eval_results(scores, df):  
    
   missclassified_probes = {}
   df_chunky = df[df.chunky == 1]
   
   for model, data in scores.items():
       for item in tqdm(data):
            # Initialize empty lists to store missclassified probes
            missclassified_non_chunky_probes = []
            missclassified_chunky_probes = []
        
            # Extract dataframe with cluster assignments
            cluster_df = data[item][0]
            cluster_df['chunky'] = df.chunky
            cluster_df['porportion'] = df.proportion
            # Extract dataframe with chunky probes
            chunky_df = cluster_df.loc[df_chunky.index]
    
            # Get clusters assigned to chunky probes
            chunky_clusters = chunky_df['cluster'].unique()
    
            # Get clusters assigned to non-chunky probes
            non_chunky_clusters = cluster_df.loc[~cluster_df.index.isin(chunky_df.index), 'cluster'].unique()
    
    
            # Extract missclassified probes in chunky cluster
            non_chunky_probes_in_chunky = cluster_df.loc[(cluster_df['chunky'] == 0 ) & (cluster_df['cluster'].isin(chunky_clusters))]
            # Extract missclassified probes in non-chunky cluster
            chunky_probes_in_non_chunky = cluster_df.loc[(cluster_df['chunky'] == 1 ) & (cluster_df['cluster'].isin(non_chunky_clusters))]
    
            # If missclassified non-chunky probes are present in chunky clusters, store them
            if not non_chunky_probes_in_chunky.empty:
                missclassified_non_chunky_probes.append(non_chunky_probes_in_chunky)
    
    
            # If chunky probes are present in non-chunky clusters, store them
            
            if not chunky_probes_in_non_chunky.empty:
                missclassified_chunky_probes.append(chunky_probes_in_non_chunky)
    
            # Save missclassified probes for the current item
            missclassified_probes[item] = {'non_chunky_probes': pd.concat(missclassified_non_chunky_probes),
                                           'chunky_probes': pd.concat(missclassified_chunky_probes)}
   return  missclassified_probes  


def get_best_model(missclassified_probes, df):
     
    min_error_rate = float('inf')
    best_item = None
    
    for item, values in missclassified_probes.items():
        non_chunky_probes = values['non_chunky_probes']
        chunky_probes = values['chunky_probes']
        # Calculate error rate
        error_rate = (len(non_chunky_probes) + len(chunky_probes)) / len(df)
        
        # Update best item if error rate is lower
        if error_rate < min_error_rate:
            min_error_rate = error_rate
            best_item = item
    return best_item, min_error_rate

def bootstrap(df, best_hyperparams, iteration, with_v_perfect = False):
   
    if with_v_perfect:
        v_perfect = df[df.Probenbezeichnung == 'V_perfect']
        df = df.drop(index = v_perfect.index)
   
    bootstrap_result = {}

    for bt in tqdm(range(iteration)):

        chunky_rows = df[df['chunky'] == 1]
        chunky_sample = chunky_rows.sample(n=max(4, len(chunky_rows)), replace=True)
        non_chunky_sample = df[df['chunky'] == 0].sample(n=len(df) - len(chunky_sample), replace=True) 
        bootstrap_sample = pd.concat([chunky_sample, non_chunky_sample], axis=0)  
        
        if with_v_perfect:
            bootstrap_sample = bootstrap_sample.append(v_perfect)
        
        bootstrap_data = bootstrap_sample.iloc[:, :-4]
        X = StandardScaler().fit_transform(bootstrap_data)
        
        
        best_model = KMeans(**best_hyperparams)
        
        # Fit the model
        best_model.fit(X)

        labels = best_model.labels_

        # Calculate the cluster centers
        centers = []
        for label in np.unique(labels):
            if label == -1:
                continue
            centers.append(np.mean(X[labels == label], axis=0))

        # Calculate the distance of each probe to its cluster center
        distances = []
        for i, probe in enumerate(X):
            if labels[i] == -1: 
                # If the probe is not in a cluster, set its distance to NaN
                distances.append(np.nan)
            else:
                distances.append(np.linalg.norm(probe - centers[labels[i]]))
        
        # Calculate the pairwise distances between the cluster centers
        cluster_distances = pdist(centers)

        # Convert the pairwise distances to a square matrix
        cluster_distances = squareform(cluster_distances)
        # np.quantile(cluster_distances , 0.5)
        
        proportions = (np.array(distances) - np.nanmin(distances)) / (np.nanmax(distances) - np.nanmin(distances))
        bootstrap_sample['distance_to_own_cluster'] = proportions
        bootstrap_sample['cluster'] = labels
        
        majority_cluster = bootstrap_sample.cluster.value_counts().idxmax()
        distances_to_majority = pd.Series(cluster_distances[majority_cluster])
        bootstrap_sample['distance_to_majority'] = bootstrap_sample['cluster'].map(distances_to_majority)
        
        if with_v_perfect:
            v_perfect_cluster = bootstrap_sample.loc[bootstrap_sample['Probenbezeichnung'] == 'V_perfect'].cluster[0]
            distances_to_perfect = pd.Series(cluster_distances[v_perfect_cluster])
            bootstrap_sample['distance_to_perfect'] = bootstrap_sample['cluster'].map(distances_to_perfect)
            
        bootstrap_result[f"bootstrap_{bt}"] = (bootstrap_sample)
    
    return bootstrap_result


def eval_bootstrap(bootstrap_result, df):
    

    df_chunky = df[df.chunky == 1] 
    misclustering_probes = {}  
    
    for item, data in bootstrap_result.items():
        
        # Initialize empty lists to store missclassified probes
        missclassified_non_chunky_probes = []
        missclassified_chunky_probes = []
         
        # Extract dataframe with cluster assignments
        cluster_df = data
        common_probes = set(df_chunky.index).intersection(set(cluster_df.index))
        # Extract dataframe with chunky probes
        chunky_df = cluster_df.loc[common_probes]
         
        # Get clusters assigned to chunky probes
        chunky_clusters = chunky_df['cluster'].unique()
         
        # Get clusters assigned to non-chunky probes
        non_chunky_clusters = cluster_df.loc[~cluster_df.index.isin(chunky_df.index), 'cluster'].unique()
         
         
        # Extract missclassified probes in chunky cluster
        non_chunky_probes_in_chunky = cluster_df.loc[(cluster_df['chunky'] == 0 ) & (cluster_df['cluster'].isin(chunky_clusters))]
        # Extract missclassified probes in non-chunky cluster
        chunky_probes_in_non_chunky = cluster_df.loc[(cluster_df['chunky'] == 1 ) & (cluster_df['cluster'].isin(non_chunky_clusters))]
         
        # If missclassified non-chunky probes are present in chunky clusters, store them
        if not non_chunky_probes_in_chunky.empty:
            missclassified_non_chunky_probes.append(non_chunky_probes_in_chunky)
         
         
        # If chunky probes are present in non-chunky clusters, store them    
        if not chunky_probes_in_non_chunky.empty:
            missclassified_chunky_probes.append(chunky_probes_in_non_chunky)
         
        # Save missclassified probes for the current item
        if missclassified_non_chunky_probes and missclassified_chunky_probes:
            misclustering_probes[item] = {'non_chunky_probes': pd.concat(missclassified_non_chunky_probes),
                                                             'chunky_probes': pd.concat(missclassified_chunky_probes)}
        elif missclassified_non_chunky_probes:
            misclustering_probes[item] = {'non_chunky_probes': pd.concat(missclassified_non_chunky_probes),
                                                             'chunky_probes': pd.DataFrame()}
        elif missclassified_chunky_probes:
            misclustering_probes[item] = {'non_chunky_probes': pd.DataFrame(),
                                                             'chunky_probes': pd.concat(missclassified_chunky_probes)}
        else:
            misclustering_probes[item] = {'non_chunky_probes': pd.DataFrame(),
                                                         'chunky_probes': pd.DataFrame()}

    return misclustering_probes

def avg_error_rate(misclustering_probes, df):
    total_error_rate = 0
    for item, values in misclustering_probes.items():
        non_chunky_probes = values['non_chunky_probes']
        chunky_probes = values['chunky_probes']
        # Calculate error rate
        error_rate = (len(non_chunky_probes) + len(chunky_probes)) / len(df)
        
        # Add error rate to total
        total_error_rate += error_rate

    # Calculate average error rate
    avg_error_rate = total_error_rate / len(misclustering_probes)
    return avg_error_rate


def creat_plots(df, target_path, target = 'distance_to_majority', target_spelling = "majority cluster"):
    
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    
    only_nonchunky = df[df.chunky == 0].sort_values(target)
    only_chunky = df[df.chunky == 1].sort_values(target)

                             
    
    #Plot 1
    plt.figure(figsize=(10, 8))
    ax = sns.boxplot(data=df, y=target, x='chunky')
    plt.title(f'Boxplot of chunky and non-chunky probes to {target_spelling}')
    plt.xlabel('Probes')
    plt.ylabel(f'Distance to {target_spelling}')
    ax.set_xticklabels(['Non-Chunky', 'Chunky'])
    plt.tight_layout()
    plt.savefig(f'{target_path}/boxplot_to_{target}')
    plt.show()
    
    #Plot 1
    plt.figure(figsize=(10, 8))
    ax = sns.boxplot(data=df, y=target)
    plt.title(f'Boxplot of all probes to {target_spelling}')
    plt.xlabel('Probes')
    plt.ylabel(f'Distance to {target_spelling}')
    plt.tight_layout()
    plt.savefig(f'{target_path}/boxplot_to_{target}_overall')
    plt.show()
    
    #Plot 2
    plt.figure(figsize=(10, 8))
    sns.barplot(data=only_chunky, y='new_name', x=target)
    plt.title(f'Distance of chunky probes to {target_spelling}')
    plt.ylabel('Probes')
    plt.xlabel(f'Distance to {target_spelling}')
    plt.tight_layout()
    plt.savefig(f'{target_path}/distance_to_{target}_chunky')
    plt.show()
    
    #Plot 2b
    plt.figure(figsize=(12, 15))
    sns.barplot(data=only_nonchunky, y='new_name', x=target)
    plt.title(f'Distance of non-chunky probes to {target_spelling}')
    plt.ylabel('Probes')
    plt.xlabel(f'Distance to {target_spelling}')
    plt.tight_layout()
    plt.savefig(f'{target_path}/distance_to_{target}_non_chunky')
    plt.show()
    
    #Plot 3
    plt.figure(figsize=(10, 8))
    sns.distplot(only_nonchunky, x = only_nonchunky[target], color = 'blue', label='Non-chunky')
    sns.distplot(only_chunky, x = only_chunky[target], color = 'orange', label='Chunky')
    
    mean_nonchunky = only_nonchunky[target].mean()
    median_nonchunky = only_nonchunky[target].median()
    plt.axvline(mean_nonchunky, color='k', linestyle='--', label='mean of non-chunky probes')
    plt.axvline(median_nonchunky, color='r', linestyle='--', label='median of non-chunky probes')
    mean_chunky = only_chunky[target].mean()
    median_chunky = only_chunky[target].median()
    plt.axvline(mean_chunky, color='green', linestyle='--', label='mean of chunky probes')
    plt.axvline(median_chunky, color='purple', linestyle='--', label='median of chunky probes')
    
    plt.title('Distribution of non-chunky and chunky probes')
    plt.xlabel(f'Distance to {target_spelling}')
    if target == 'distance_to_perfect':
        plt.legend(loc='upper left')
    else:
        plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'{target_path}/distplot_to_{target}')
    plt.show()
    

def get_new_proportion(df, target):
# Separate the chunky and non-chunky probes

    scaler = MinMaxScaler(feature_range=(0.01, 0.99))
    
    threshold = round(np.quantile(df[target], 0.75), 2)
    df.loc[df[target] < threshold, 'new_proportion'] = 0
    df.loc[df[target] >= threshold, 'new_proportion'] = scaler.fit_transform(df.loc[df[target] >= threshold, [target]])
    return df



def run(path, target_path, iteration = 2000, with_v_perfect = False):
    
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    
    
    df = pd.read_csv(path)
    
    if not with_v_perfect:
        
        v_perfect = df[df.Probenbezeichnung == 'V_perfect']
        df = df.drop(index = v_perfect.index)
        


    hyperparameter_scores = run_hyperparamter_grid_model(df)
    
    with open(f'{target_path}/hyperparameter_scores_with_v_{with_v_perfect}', 'wb') as f:
        pickle.dump(hyperparameter_scores, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    eval_scores = eval_results(hyperparameter_scores, df)
    best_model, min_error_rate = get_best_model(eval_scores, df)
    model_name = next(iter(hyperparameter_scores))
        
    # print(f"Best model: {best_model}, Error rate: {min_error_rate * 100:.2f}%")
    
    with open(f'{target_path}/best_model_with_v_{with_v_perfect}.txt', 'w') as f:
        print(f"Best model: {best_model}, Error rate: {min_error_rate * 100:.2f}%", file=f)
    
    best_hyperparams = hyperparameter_scores[model_name][best_model][1]
       
    bootstrap_result =  bootstrap(df, best_hyperparams = best_hyperparams, iteration=iteration, with_v_perfect=with_v_perfect)
    

    with open(f'{target_path}/bootstrap_result_{iteration}_with_v_{with_v_perfect}', 'wb') as f:
        pickle.dump(bootstrap_result, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    missclustering_boostrap = eval_bootstrap(bootstrap_result, df)
    error_rate = avg_error_rate(missclustering_boostrap, df)
    with open(f'{target_path}/best_model_with_v_{with_v_perfect}_bootstrap_avg_error.txt', 'w') as f:
        print(f'Average error rate over {iteration} iterations of bootstrap: {error_rate * 100:.2f}%', file=f)
        
    # print(f'Average error rate over {iteration} iterations of bootstrap: {error_rate * 100:.2f}%')
    
    bootstrap_results_df = pd.concat(bootstrap_result, axis=0)
    
    if with_v_perfect:
        grouped_perfect = bootstrap_results_df.groupby('Probenbezeichnung')['distance_to_perfect', 'chunky'].mean().reset_index()
        grouped_perfect = grouped_perfect.merge(df[['Probenbezeichnung', 'proportion']], on='Probenbezeichnung', how='left')
        grouped_perfect['new_name'] =  grouped_perfect['Probenbezeichnung'] + ' (' + grouped_perfect['proportion'].astype(str) + ')'
        
        creat_plots(grouped_perfect, target_path = f'{target_path}/perfect', target = 'distance_to_perfect', target_spelling = 'perfect probe cluster')
        
        
    grouped = bootstrap_results_df.groupby('Probenbezeichnung')['distance_to_majority', 'chunky'].mean().reset_index()
    grouped = grouped.merge(df[['Probenbezeichnung', 'proportion']], on='Probenbezeichnung', how='left')
    grouped['new_name'] =  grouped['Probenbezeichnung'] + ' (' + grouped['proportion'].astype(str) + ')'
        
    creat_plots(grouped, target_path = f'{target_path}/majority/with_v_perfect_{with_v_perfect}')
    
    #distance_to_majority is better than distance_to_perfect
    new_prop = get_new_proportion(grouped, target = 'distance_to_majority')
    final_df = df.merge(new_prop[['Probenbezeichnung', 'new_proportion']], on='Probenbezeichnung', how='left')
    
    final_df.to_csv(f'{target_path}/final_df_with_v_{with_v_perfect}.csv')
    


#add location of dataframes df_quantiles.csv  
paths = ['XXX']

#add target location
target_paths = ['XXXX']

for i in range(len(paths)):
    run(path = paths[i], target_path= target_paths[i])
    
    if i == 0:
        run(path = paths[i], target_path= target_paths[i], with_v_perfect=True)
