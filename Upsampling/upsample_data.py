# Import necessary libraries
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from scipy.stats import gaussian_kde


# Main function
def upsample(input_path, upsampling_max, num_random_state, output_path):

    # Helper function
    def generate_samples(times_upsampling_chunky, times_upsampling_nonchunky, random_state):

        # Load all CSV files into separate dataframes
        df_feret = pd.read_csv(f"{input_path}/feret_data.csv")
        df_graphite = pd.read_csv(f"{input_path}/graphite_area_data.csv")
        df_compact = pd.read_csv(f"{input_path}/compact_data.csv")
        df_convexity = pd.read_csv(f"{input_path}/convexity_data.csv")
        df_nn_distance = pd.read_csv(f"{input_path}/nn_distance_data.csv")
        df_roundness = pd.read_csv(f"{input_path}/roundness_data.csv")
        df_sphericity = pd.read_csv(f"{input_path}/sphericity_data.csv")

        # Combine all dataframes into a list of tuples to loop through later
        dataframes = [(df_feret, "feret"), (df_graphite, "graphite"), (df_compact, "compact"),
                      (df_convexity, "convexity"),
                      (df_nn_distance, "nn_distance"), (df_roundness, "roundness"), (df_sphericity, "sphericity")]

        train_dataframes = []
        upsampled_dataframes = []

        # Loop through each dataframe and prepare the data for training
        for df in dataframes:
            temp_df = df[0]
            temp_name = df[1]

            # Remove first row (header) from each dataframe
            temp_df = temp_df.iloc[1:]

            # Insert a column called "proportion" with the specific values
            temp_df.insert(1, "proportion",
                           [0, 0, 0, 0, 0, 0, 0, 0, 0.7, 0, 0, 0, 0.01, 0, 0.3, 0, 0.075, 0, 0, 0, 0, 0, 0, 0.185, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0.6, 0.365, 0.807, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0])

            # Split the data into training and testing sets
            train, test = train_test_split(temp_df, train_size=0.7, stratify=temp_df['chunky'],
                                           random_state=random_state)
            train_dataframes.append((train, test, temp_name))

        # Loop through the specified number of times to upsample the "chunky" samples
        for n in range(times_upsampling_chunky):

            # Get all the row indices from the first dataframe
            rows = train_dataframes[0][0].iloc[:, 0]

            # Remove the last n * 54 rows (where n is the current iteration number)
            rows = rows[1:len(rows) - n * 54]

            # skip samples which are non-chunky in this loop
            for sample in rows.keys():
                if train_dataframes[0][0].loc[sample]["chunky"] == 'N':
                    continue

                # get current proportion and current sample_id
                for_proportion = train_dataframes[0][0].loc[sample]
                cur_proportion = for_proportion[1]
                cur_probenbezeichnung = for_proportion[0]
                for_proportion = for_proportion[3:-1]
                for_proportion = for_proportion.dropna()

                # initialize the array for the multivariate kde
                dim_array = np.zeros((7,len(for_proportion)))
                for i,df in enumerate(train_dataframes):
                    one_probe = df[0].loc[sample]
                    one_probe = one_probe[3:-1]
                    one_probe = one_probe.dropna()
                    one_probe = one_probe.to_numpy(np.float64)
                    dim_array[i] = one_probe

                # estimate the pdf for the current sample
                ecdf = gaussian_kde(dim_array,bw_method="scott")

                # draw values from the estimated pdf
                x = ecdf.resample(len(for_proportion)).squeeze().tolist()

                # fill missing values with NaNs if necessary
                fill_nas = [np.NaN for i in range(8650-len(x[0]))]

                # Add remaining attributes
                for i in range(7):
                    x[i].extend(fill_nas)
                    x[i].insert(0, pd.NA)
                    x[i].insert(0,cur_proportion)
                    x[i].insert(0, f"V_sample_from_{cur_probenbezeichnung}_No_{n+1}")
                    x[i].append('Y')

                # append dataframe to the list of upsampled dataframes
                for i in range(7):
                    upsampled_dataframes.append((x[i],train_dataframes[i][2]))
      
        # same loop for the non-chunky samples
        for n in range(times_upsampling_nonchunky):

            # Get all the row indices from the first dataframe
            rows = train_dataframes[0][0].iloc[:,0]

            # Remove the last n * 54 rows (where n is the current iteration number)
            rows = rows[1:len(rows)-n*54]

            # skip samples which are chunky in this loop
            for sample in rows.keys():
                if train_dataframes[0][0].loc[sample]["chunky"] == 'Y':
                    continue

                # get current proportion and current sample_id
                for_proportion = train_dataframes[0][0].loc[sample]
                cur_proportion = for_proportion[1]
                cur_probenbezeichnung = for_proportion[0]
                for_proportion = for_proportion[3:-1]
                for_proportion = for_proportion.dropna()

                # initialize the array for the multivariate kde
                dim_array = np.zeros((7,len(for_proportion)))
                for i,df in enumerate(train_dataframes):
                    one_probe = df[0].loc[sample]
                    one_probe = one_probe[3:-1]
                    one_probe = one_probe.dropna()
                    one_probe = one_probe.to_numpy(np.float64)
                    dim_array[i] = one_probe

                # estimate the pdf for the current sample
                ecdf = gaussian_kde(dim_array,bw_method="scott")

                # draw values from the estimated pdf
                x = ecdf.resample(len(for_proportion)).squeeze().tolist()

                # fill missing values with NaNs if necessary
                fill_nas = [np.NaN for i in range(8650-len(x[0]))]

                # Add remaining attributes
                for i in range(7):
                    x[i].extend(fill_nas)
                    x[i].insert(0, pd.NA)
                    x[i].insert(0,cur_proportion)
                    x[i].insert(0, f"V_sample_from_{cur_probenbezeichnung}_No_{n+1}")
                    x[i].append('Y')

                # append dataframe to the list of upsampled dataframes
                for i in range(7):
                    upsampled_dataframes.append((x[i],train_dataframes[i][2]))

        # make a list for the final dataframes
        final_dfs = []

        # iterate over training dataframes
        for i in range(7):
            final = train_dataframes[i][0]
            final.reset_index(drop=True, inplace=True)
            print(f"Working on Dataframe {i}")

            # extend training set with new samples
            for j in range(len(upsampled_dataframes)//7):
                assert train_dataframes[i][2] == upsampled_dataframes[j*7+i][1]
                final.loc[len(final)] = upsampled_dataframes[j*7+i][0]

            # append current training dataframe with upsampled samples to list of final data frames
            final_dfs.append((pd.concat([final,train_dataframes[i][1]]),train_dataframes[i][2]))
    

        # make folders corresponding to current upsampling ratio and random state
        if not os.path.exists(f"{output_path}/times_upsampling_nonchunky_{times_upsampling_nonchunky}-times_upsampling_chunky_{times_upsampling_chunky}"):
            os.mkdir(f"{output_path}/times_upsampling_nonchunky_{times_upsampling_nonchunky}-times_upsampling_chunky_{times_upsampling_chunky}")
        os.chdir(f"{output_path}/times_upsampling_nonchunky_{times_upsampling_nonchunky}-times_upsampling_chunky_{times_upsampling_chunky}")
        if not os.path.exists(f"{output_path}/times_upsampling_nonchunky_{times_upsampling_nonchunky}-times_upsampling_chunky_{times_upsampling_chunky}/random_state_{random_state}"):
            os.mkdir(f"{output_path}/times_upsampling_nonchunky_{times_upsampling_nonchunky}-times_upsampling_chunky_{times_upsampling_chunky}/random_state_{random_state}")
        os.chdir(f"{output_path}/times_upsampling_nonchunky_{times_upsampling_nonchunky}-times_upsampling_chunky_{times_upsampling_chunky}/random_state_{random_state}")

        # save the data frames in the newly created folder
        for df in final_dfs:
            temp_df = df[0]
            temp_df.reset_index(drop=True, inplace=True)
            temp_df.to_csv(f"{output_path}/times_upsampling_nonchunky_{times_upsampling_nonchunky}-times_upsampling_chunky_{times_upsampling_chunky}/random_state_{random_state}/df_{df[1]}_upsampled.csv", index=False)
    
    # call the helper function according to the input of the upsample function
    for a in range(upsampling_max):
        for b in range(a, upsampling_max):
            for i in range(num_random_state):
                generate_samples(b, a, random_state=i)

# call the function
upsample(input_path="/path/to/csvs", upsampling_max=5, num_random_state=7, output_path="/path/to/output/directory")
