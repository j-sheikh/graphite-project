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

Results Results Results Results Results Results Results Results Results Results Results Results Results Results Results 
Results Results Results Results Results Results Results Results Results Results Results Results Results Results Results 
 Results Results Results Results Results Results Results Results Results Results Results Results Results Results Results 
 Results Results Results Results Results Results Results Results Results Results Results Results Results Results Results 
 Results Results Results Results Results Results Results Results Results Results Results Results Results Results Results 


## Prerequisites
To use our methods the follwing prerequisites need to be forfilled:  <br>


**Software**
Python<br>
Excell<br>
...<br>


**Data Format**
<br>

## Workflow
<p align="center">
  <img src='https://github.com/IZMEHD/graphite-project/blob/main/Images/Workflow.JPG' width=550 > 
</p>
