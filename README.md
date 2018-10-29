# statsmodels_contribution

Linear Mixed Effects Modelling in Python.   
Contribution to the statsmodels package https://github.com/statsmodels/statsmodels by Mathilde Duverger and Marie Ivchtchenko.

## Linear Mixed Effects Modelling

Statsmodels allows to use a linear mixed effects model for regression for dependent data.  
To see how to create and fit your lme model see the documentation here: https://www.statsmodels.org/stable/mixed_linear.html  
The predict method of the statsmodels Linear Mixed Effect Model only predict based on the fixed effects... which loses the interest of linear mixed effects modelling !  
To this end, we have created the PredictMixed class to help you predict the fixed and random effects for grouped data.

## PredictMixed class

The PredictMixed class takes as instantiation input the model created and fit by statsmodels mixedlm.  
To fit the mixedlm to a particular group one should use the fit_re method. The random effects fitted in this method can then be used by calling the predict_re method.  

The algebra used to calculate the random effects are derived from the Statistics in Action with R class from Ecole Polytechnique in Spring 2018 (http://sia.webpopix.org/lme.html) taught by Marc Lavielle.




