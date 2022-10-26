# SYSC-5405 - Pattern Classification and Experiment Design
Pattern Classification and Experiment Design - Classification of Protein Methylation Using Probabilistic Neural Networks

Probabilistic Neural Networks (PNNs) utilize Bayesian classification and Parzen windows to smooth the data's class-likelihoods. They facilitate online learning by allowing new patterns to be simply normalized and added as a new node to the pattern layer. PNNs were used to predict protein methylation for the purposes of this project. The task was to predict whether a protein window was methylated given their feature data, a set of 28 features were from ProtDCal. In the following, we will go over our experiment design and how we used PNN to predict protein methylation. 

Note: unless otherwise specified, all scores in this experiment are estimated using stratified 10-fold cross-validation.
