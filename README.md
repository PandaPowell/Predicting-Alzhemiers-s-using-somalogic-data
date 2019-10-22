# Predicting-Alzhemiers-s-using-somalogic-data

The data consists of a SOMAscan assay for quantifying 1001 proteins in blood samples. 
It was obtained from AddNeuroMed a king’s cohort study into dementia. 
Clinical diagnosis were provided for 331 Alzheimer’s pateints, 211 controls, and 149 mild cognitive impaired subjects. 
I optimized a random forest approach to create a model able to predict Alzheimer’s diagnosis based on a small number of 
protein levels in blood to an accuracy of around 70%, the analysis was based on Sattlecker’s analysis who was formerly 
at king’s.  

The goal was to optimize the model further by selecting the most predictive proteins are predictiors, 
as this analysis is intended for prospective clinical application, it is desirable to keep the number of proteins 
as low as possible while maintaining a high performance. Hyper parameter optimization was performed using 
Scikit-Learn’s RandomizedSearchCV method, we can define a grid of hyperparameter ranges, and randomly sample from the grid, 
performing K-Fold CV with each combination of values.

The model was optimised to an average of 70% accuracy with the optimal number of proteins being 13. 
Below are the 13 proteins ranked by importance.

                 importance
PSA.ACT            0.123709
SEPR               0.111421
CK.MB              0.097687
SDF.1b             0.092230
Apo.A.I            0.087772
IGFBP.2            0.082726
Trypsin            0.071976
RBP                0.064789
Myeloperoxidase    0.060251
IL24               0.055419
CASA               0.054263
ABL1               0.052458
Protein.C          0.045301

- Explain plan for further optimization.
My analysis ended here but if further optimisation of the model was planned I would perform backwards elimination 
at multiple different stages of feature selection to arrive at the top 13 proteins.
