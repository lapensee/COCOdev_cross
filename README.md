### COCO model code  
The repository contains code for the COCO model (Osth and Dennis, 2015) used in Yim et al. (2020) Sources of Interference in Memory across Development (project page on OSF [https://osf.io/2fg5w/](https://osf.io/2fg5w/) )

### Required libraries
* The code has been tested on Ubuntu 18.04 with python 2.7.13 and using the following libraries
* matplotlib (2.0.0), numpy (1.13.3), pandas (0.20.3), scipy (1.0.0), seaborn (0.8.1), statsmodels (0.8.0)

### Usage
*cocoFit.py* is the main file. Go to line 1596 to change any settings for the model fitting parameters, which includes:
* DE-MCMC settings (currently set as described in the paper)
* fit function and prediction function for adults, 7-8yrs, 4-5yrs
* plotting posterior distributions, model fits, decomposed noises