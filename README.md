# Utilizing Particle Property Distributions to Determine Chunky Graphite Ratios in Iron Castings.

***Summary*** 


This project demonstrates new ap-proaches in determining the ratio of degenerated or “chunky” graphite in casing samples.
By using the distribution of particle parame-ters (roundness, compactness, convexity, i.e) and applying on that multivariate adaptive regression splines (MARS), 
generalized additive models (GAM) and clustering methods we show that it is possible to determine the chunky
graphite-ratio with higher accuracy than with traditional approaches that focus on single particles to determine their type. Additionally we demonstrate an up-sampling method to account
for sparse data and to further increase accuracy on chunky graphite-ratio prediction.


### Table of Contents
**[Motivation](#[Motivation])**<br>
**[Results](#Results)**<br>
**[Prerequisites](#prerequisites)**<br>
**[Workflow](#setup)**<br>


## Motivation

In iron casting degenerated graphite, or chunky graphite can have an undesirable effect on mechanical properties like ductility.
 Therefore it is important to identify castings that contain chunky graphite to ensure the quality and performance of a given cast or for a type of alloy.
 A common approach to determine the presence of chunky graphite is to take a picture of a cross section of a sample and then use image processing to identify chunky particles.
 Using the roundness of individual particles, or additional parameters like compactness or convexity as indicators of degeneration.
 These criteria applied to individual particles are not sufficient however to identify all chunky particles.
 


<p align="center">
  <img src='https://github.com/IZMEHD/graphite-project/blob/main/Images/ChunkyEverywhere.JPG' width=550 > 
</p>

*Even when using multivariate classification of particle type we find by manual inspection that chunky graphite is present in all types.*


This lack in precision of identification of degenerated graphite can lead to wrong assumptions about material property and allows for unaccounted variation in performance of castings.
Considering this, we set out to find methods that can give a more precise representation of the chunky-graphite-ratio in a given sample. 
For this we investigated multiple statistical methods for their effectiveness to solve this task. 
All of them utilise the distribution of particle properties of a given sample instead of focusing on individual par- ticles. In the following we discuss our approaches and evaluate their performance.


## Results

**MARS & GAM** <br>

Four variants of the MARS model are used -  trained on the original dataset of 55 observations for predicting original proportions and new proportions , another trained on an up-sampled dataset of 109 observations for predicting original proportions and new proportions. MARS model trained for predicting original proportions, the one trained on the original dataset performs better than the one trained on the up-sampled dataset. The RMSE is 0.0602 for the model trained on the original dataset, which is lower than the model trained on the up-sampled dataset having RMSE of 0.0952. The R2 value of 0.9332 is also higher than the model trained on the up-sampled dataset having R2 value of 0.8334.

We used the same methods for predicting new proportions generated through the clustering approach. And found that both versions of the MARS model(trained on original data and up-sampled data) perform exceptionally well on test data for predicting new proportions. The RMSE values for the model trained on original data and upsampled data are 0.0318 and 0.0271 respectively and R2 value are 0.9862 and 0.9899 for model trained on original data and upsampled data respectively. 

However, the model predicts non-positive values for some of the non-degenerate graphite samples in MARS model, this can be remedied with the GAM model.

GAM is also trained in the same manner like MARS model having 4 different version two for predicting original proportions and two for predicting new proportions.The GAM model trained on the original dataset has almost identical predictions to the MARS model trained on the original dataset, with an R-squared value of 0.9596 and an RMSE value of 0.0469 on the test data. Meanwhile, the GAM model trained on the up-sampled data still has similar predictions to the MARS model. Additionally, all predicted values are in the range of 0 to 1, which is an advantage over the MARS model because the GAM model can handle a variety of response distributions.

![image](https://user-images.githubusercontent.com/103118176/229027050-8b14163a-bd79-4097-be27-b1ad1961e231.png)

To summarize, after looking at the predicted values and RMSE and R2 values from the image above, it is evident that the GAM model trained on the original dataset performs better than the GAM model trained on the up-sampled dataset, similar to the MARS model. Therefore, up-sampling does not significantly improve predictions for original proportions.

Similarly, we trained two versions of the GAM model for predicting new proportions. However, neither of the models provided as accurate predictions as the MARS model, and the RMSE and R2 values for these models are also not as good compared to all other models.


**Clustering** <br>

Our clustering approach allows us to come up with a way to determine the relative degenerated graphite ratio without using
percentage labels, thus enabling predictions without the the need for them. Clustering also offers a
possible metric to determine how similar samples are.



## Prerequisites
To use our methods the follwing prerequisites need to be forfilled:  <br>


**Software** <br>
Python<br>
Excell<br>
...<br>

Before running the code, make sure you have the following packages installed in your R environment:

rsample, caret, vip, pdp, Metrics, dplyr, glmnet, mgcv, caTools, segmented


**Data Format**
<br>

## Workflow
<p align="center">
  <img src='https://github.com/IZMEHD/graphite-project/blob/main/Images/Workflow.JPG' width=550 > 
</p>


The following is meant as an overview and as in-
struction for the workflow to obtain estimates of
new samples.

**Aggregating data - CSV-to-Excel**
The first step
in the process is to aggregate the measurements of
multiple samples into one location. For this the
provided program (https://www.dropbox.com/s/u4s936829du74rr/sisyphus_0.1.0.zip?dl=0) can be
used. The program scans a selected folder and all
sub-folders for measurement .csv files and merges
them into one Excel .xlsx file. All .csv in the selected folder must be of the source data type. 
The names of the csv. files must have this pattern:<br>

([any text exept "\_" ]\_[any text exept "\_"]\_[Sample groupID as number]\_[Sample id in group as number].csv)<br>

so for example:<br>
Sample 1 from group 50: ABC_XYZ_50_1.csv<br>
Sample 2 from group 50: ABC_XYZ_50_2.csv<br>

The program uses a .xlsx template. Do not delete the template and keep it in the same folder as the .exe of the program. 


 

**Categorizing data - Excel-to-CSV and Transform Data-to-Quantile** 

Split the data of all samples into individual
csv. files for each of the particle attributes. After
this step we have 7 files, each representing one
attribute for all samples and there particles. After that
the distribution quantiles are generated from the attribute
data. 
The __main__.py needs just the location of the files and will execute both steps,
by calling the scripts read_graphite_data.py in Excel to CSV and transform_data.py in Transform Data to Quantile 

**Upsampling** 
As discussed in Section 3 it is pos-
sible to upsample the data to improve prediction
performance. For this the user needs to provide the
path to the csv files generated by transformdata-
toquantile.py as input, the number of how many
times the data should be upsampled at most and an
output path.

**Transform Data-to-Quantile** 
In this step the dis-
tribution quantiles are generated from the attribute
data. This Python script needs the location of the
attribute data as input.

**CLUSTERING** 
As shown in Section 4, cluster-
ing is a promising approach to distinguish between
non-chunky and chunky probes as well as deter-
mining a proportion of chunkiness, just from the
data itself. The user needs to provide the path to
data frame of interest, df quantiles.

MARS & GAM 
This repository contains an R code that predicts the proportion of degenerated graphite in a dataset using various models. 
The saved models are MARS_or.rds, MARS_up.rds, GAM_or.rds, and GAM_up.rds , MARS_or_np.rds, MARS_up_np.rds, GAMor_np.rds, and GAMup_np.rds
You can use these saved models to make predictions on your own dataset by loading them into your R environment and running the prediction function in the model_for_prediction.r file.

Load the saved model you want to use into your R environment using the readRDS() function. The saved models are located in the models directory.
Load your dataset using the read.csv() function.
Call the prediction function with the loaded model and dataset paths as arguments. The function returns a list of prediction table, RMSE, and R-squared value.
Print the results.

