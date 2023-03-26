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

**MARS & GAN**<br>

The MARS model was trained on an up-sampled dataset with 87 observations, which se-
lected 9 out of 11 terms and 7 out of 42 predictors using the nprune parameter set to 10. The model
demonstrated that the most important predictors
are ’graphite distance to total mean’, ’convexity
distance to 75 quantile’, ’nn 25 quantile’, and other
significant predictors.  <br>

The up-sampled MARS model’s performance
was evaluated again using RMSE and R-squared
values on the test dataset, with the RMSE value
being 0.00718 and the R2 value being 0.998. Boot-
strapping was also utilized to evaluate the model’s
accuracy, with the mean of the predictions being
0.0622 after 10,000 iterations. The up-sampled
MARS model performed exceptionally well on
the unseen dataset hence overcomed the problem
of overfitting. <br>



THe GAM model trained on the up-sampled dataset
performed mostly well, with an R-squared value
of 0.995 and promising predictions. 
The model had an MSE and R- squared value on the test data of
0.0414 and 0.940 respectively.

**Clustering **<br>
Our clustering approach allows us to come up with a way to determine the relative degenerated graphite ratio without using
percentage labels, thus enabling predictions without the the need for them. Clustering also offers a
possible metric to determine how similar samples are.



## Prerequisites
To use our methods the follwing prerequisites need to be forfilled:  <br>


**Software**<br>
Python<br>
Excell<br>
...<br>


**Data Format**
<br>

## Workflow
<p align="center">
  <img src='https://github.com/IZMEHD/graphite-project/blob/main/Images/Workflow.JPG' width=550 > 
</p>
