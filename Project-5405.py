#!/usr/bin/env python
# coding: utf-8

# In[2]:


#--------------------------------------------------------------------------------
#
# Project implementation.
#
#--------------------------------------------------------------------------------
# !pip install neupy
# !pip install --upgrade mlxtend
import pandas as pd
from scipy import stats
import numpy as np
from numpy import array
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_predict, StratifiedKFold,     LeaveOneOut

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score,     recall_score, roc_curve, precision_recall_curve, make_scorer

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.feature_selection import SelectFromModel
from neupy import algorithms
from neupy.utils import format_data, iters
from mlxtend.feature_selection import SequentialFeatureSelector as Sfs
from mlxtend.feature_extraction import LinearDiscriminantAnalysis,     PrincipalComponentAnalysis
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import BaggingClassifier
import os

#--------------------------------------------------------------------------------
np.random.seed(0)
# from google.colab import drive
# drive.mount('/content/gdrive')

#--------------------------------------------------------------------------------

RATIO = 4/7.
BASE_DIR = F"C:/Users/zalat/Downloads/project/"
CSV_FILE_TRAINING = F"csv_result-Descriptors_Training.csv"
CSV_FILE_CALIBRATION = F"csv_result-Descriptors_Calibration.csv"
POSITIVE_LABEL = 'P'
NEGATIVE_LABEL = 'N'
SELECTED_FEATURES = ('IP_ES_25_N1', 'Z1_IB_10_N1', 'Gs(U)_IB_12_N1',
                     'Gs(U)_IB_68_N1', 'Gs(U)_IB_58_N1', 'Gs(U)_IB_60_N1',
                     'Z1_NO_sideL35_M', 'HP_NO_sideL35_CV',
                     'Z1_NO_sideR35_CV', 'Pb_NO_sideR35_S',
                     'IP_NO_sideL35_SI71', 'Z1_NO_PRT_CV', 'Z2_NO_AHR_CV',
                     'Gs(U)_NO_ALR_SI71', 'Z3_NO_UCR_S', 'Z3_NO_UCR_N1',
                     'ECI_NO_UCR_CV', 'Pa_NO_BSR_SI71', 'ISA_NO_NPR_S',
                     'Z3_NO_NPR_V', 'IP_NO_PLR_S', 'Pb_NO_PCR_V',
                     'ECI_NO_PCR_CV')
SELECTED_FEATURES_5 = ('IP_NO_PLR_S', 'ISA_NO_NPR_S', 'IP_ES_25_N1',
                       'Gs(U)_NO_ALR_SI71', 'Z1_NO_PRT_CV', 'Gs(U)_IB_58_N1',
                       'Z1_IB_5_N1', 'Gs(U)_IB_68_N1', 'Gs(U)_IB_60_N1')
# Selected using most discriminating features and LSVC.
SELECTED_FEATURES_5 = ('IP_NO_PLR_S', 'ISA_NO_NPR_S', 'IP_ES_25_N1',
                       'Gs(U)_NO_ALR_SI71', 'Z1_NO_PRT_CV', 'Gs(U)_IB_58_N1',
                       'Z1_IB_5_N1', 'Gs(U)_IB_68_N1', 'Gs(U)_IB_60_N1',
                       'Z3_NO_UCR_S', 'Z2_NO_AHR_CV')
# Selected using only infogain.
SELECTED_FEATURES_5 = ('IP_NO_PLR_S', 'ISA_NO_NPR_S', 'Z1_NO_PRT_CV',
                       'Z2_NO_AHR_CV', 'Gs(U)_NO_ALR_SI71', 'HP_NO_sideL35_CV',
                       'Z1_NO_sideR35_CV', 'Z3_IB_4_N1', 'Z1_IB_5_N1',
                       'Z1_NO_sideL35_M', 'Pb_NO_sideR35_S')
# Selected using backward elimination. (1st attempt)
SELECTED_FEATURES_6 = ('Z1_IB_10_N1', 'Z1_IB_5_N1')

# Improved backward el. (2nd attempt)
SELECTED_FEATURES_7 = ('Z1_IB_10_N1', 'Z1_IB_5_N1', 'ECI_NO_UCR_CV')

ALL_SUBSETS_47 = {28: {'feature_idx': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27), 'cv_scores': array([0.15897436, 0.00118343, 0.15686275, 0.00121507, 0.15996503,
       0.00121507, 0.16619183, 0.        , 0.07858377, 0.07067425]), 'avg_score': 0.07948655511966006, 'feature_names': ('IP_ES_25_N1', 'Z3_IB_4_N1', 'Z1_IB_10_N1', 'Z1_IB_5_N1', 'Z3_IB_8_N1', 'ECI_IB_4_N1', 'ECI_IB_5_N1', 'Gs(U)_IB_12_N1', 'Gs(U)_IB_68_N1', 'Gs(U)_IB_58_N1', 'Gs(U)_IB_60_N1', 'Z1_NO_sideL35_M', 'HP_NO_sideL35_CV', 'Z1_NO_sideR35_CV', 'Pb_NO_sideR35_S', 'IP_NO_sideL35_SI71', 'Z1_NO_PRT_CV', 'Z2_NO_AHR_CV', 'Gs(U)_NO_ALR_SI71', 'Z3_NO_UCR_S', 'Z3_NO_UCR_N1', 'ECI_NO_UCR_CV', 'Pa_NO_BSR_SI71', 'ISA_NO_NPR_S', 'Z3_NO_NPR_V', 'IP_NO_PLR_S', 'Pb_NO_PCR_V', 'ECI_NO_PCR_CV')}, 27: {'feature_idx': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27), 'cv_scores': array([0.15646941, 0.00116009, 0.16135458, 0.        , 0.16332117,
       0.00116009, 0.16233766, 0.        , 0.08268734, 0.07791096]), 'avg_score': 0.08064013031391057, 'feature_names': ('IP_ES_25_N1', 'Z3_IB_4_N1', 'Z1_IB_10_N1', 'Z1_IB_5_N1', 'Z3_IB_8_N1', 'ECI_IB_4_N1', 'ECI_IB_5_N1', 'Gs(U)_IB_12_N1', 'Gs(U)_IB_68_N1', 'Gs(U)_IB_58_N1', 'Gs(U)_IB_60_N1', 'Z1_NO_sideL35_M', 'HP_NO_sideL35_CV', 'Z1_NO_sideR35_CV', 'Pb_NO_sideR35_S', 'IP_NO_sideL35_SI71', 'Z1_NO_PRT_CV', 'Z2_NO_AHR_CV', 'Gs(U)_NO_ALR_SI71', 'Z3_NO_UCR_S', 'Z3_NO_UCR_N1', 'Pa_NO_BSR_SI71', 'ISA_NO_NPR_S', 'Z3_NO_NPR_V', 'IP_NO_PLR_S', 'Pb_NO_PCR_V', 'ECI_NO_PCR_CV')}, 26: {'feature_idx': (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27), 'cv_scores': array([0.16231884, 0.00115607, 0.161     , 0.        , 0.16263941,
       0.00117096, 0.16131989, 0.        , 0.07887818, 0.07876106]), 'avg_score': 0.08072444043294415, 'feature_names': ('IP_ES_25_N1', 'Z3_IB_4_N1', 'Z1_IB_10_N1', 'Z1_IB_5_N1', 'Z3_IB_8_N1', 'ECI_IB_4_N1', 'Gs(U)_IB_12_N1', 'Gs(U)_IB_68_N1', 'Gs(U)_IB_58_N1', 'Gs(U)_IB_60_N1', 'Z1_NO_sideL35_M', 'HP_NO_sideL35_CV', 'Z1_NO_sideR35_CV', 'Pb_NO_sideR35_S', 'IP_NO_sideL35_SI71', 'Z1_NO_PRT_CV', 'Z2_NO_AHR_CV', 'Gs(U)_NO_ALR_SI71', 'Z3_NO_UCR_S', 'Z3_NO_UCR_N1', 'Pa_NO_BSR_SI71', 'ISA_NO_NPR_S', 'Z3_NO_NPR_V', 'IP_NO_PLR_S', 'Pb_NO_PCR_V', 'ECI_NO_PCR_CV')}, 25: {'feature_idx': (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 25, 26, 27), 'cv_scores': array([0.15819751, 0.00118624, 0.16129032, 0.        , 0.16591928,
       0.00118064, 0.16074766, 0.        , 0.08093525, 0.07733333]), 'avg_score': 0.08067902381306255, 'feature_names': ('IP_ES_25_N1', 'Z3_IB_4_N1', 'Z1_IB_10_N1', 'Z1_IB_5_N1', 'Z3_IB_8_N1', 'ECI_IB_4_N1', 'Gs(U)_IB_12_N1', 'Gs(U)_IB_68_N1', 'Gs(U)_IB_58_N1', 'Gs(U)_IB_60_N1', 'Z1_NO_sideL35_M', 'HP_NO_sideL35_CV', 'Z1_NO_sideR35_CV', 'Pb_NO_sideR35_S', 'IP_NO_sideL35_SI71', 'Z1_NO_PRT_CV', 'Z2_NO_AHR_CV', 'Gs(U)_NO_ALR_SI71', 'Z3_NO_UCR_S', 'Z3_NO_UCR_N1', 'Pa_NO_BSR_SI71', 'ISA_NO_NPR_S', 'IP_NO_PLR_S', 'Pb_NO_PCR_V', 'ECI_NO_PCR_CV')}, 24: {'feature_idx': (0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 25, 26, 27), 'cv_scores': array([0.15824595, 0.00113895, 0.16079924, 0.        , 0.16849817,
       0.00115473, 0.16501353, 0.        , 0.07926829, 0.07574468]), 'avg_score': 0.08098635416486549, 'feature_names': ('IP_ES_25_N1', 'Z3_IB_4_N1', 'Z1_IB_10_N1', 'Z1_IB_5_N1', 'Z3_IB_8_N1', 'ECI_IB_4_N1', 'Gs(U)_IB_12_N1', 'Gs(U)_IB_68_N1', 'Gs(U)_IB_60_N1', 'Z1_NO_sideL35_M', 'HP_NO_sideL35_CV', 'Z1_NO_sideR35_CV', 'Pb_NO_sideR35_S', 'IP_NO_sideL35_SI71', 'Z1_NO_PRT_CV', 'Z2_NO_AHR_CV', 'Gs(U)_NO_ALR_SI71', 'Z3_NO_UCR_S', 'Z3_NO_UCR_N1', 'Pa_NO_BSR_SI71', 'ISA_NO_NPR_S', 'IP_NO_PLR_S', 'Pb_NO_PCR_V', 'ECI_NO_PCR_CV')}, 23: {'feature_idx': (0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 22, 23, 25, 26, 27), 'cv_scores': array([0.15727273, 0.00117096, 0.15950334, 0.0011976 , 0.16872038,
       0.00116414, 0.16233184, 0.        , 0.07865169, 0.07491582]), 'avg_score': 0.08049285075098568, 'feature_names': ('IP_ES_25_N1', 'Z3_IB_4_N1', 'Z1_IB_10_N1', 'Z1_IB_5_N1', 'Z3_IB_8_N1', 'ECI_IB_4_N1', 'Gs(U)_IB_12_N1', 'Gs(U)_IB_68_N1', 'Gs(U)_IB_60_N1', 'Z1_NO_sideL35_M', 'HP_NO_sideL35_CV', 'Z1_NO_sideR35_CV', 'Pb_NO_sideR35_S', 'Z1_NO_PRT_CV', 'Z2_NO_AHR_CV', 'Gs(U)_NO_ALR_SI71', 'Z3_NO_UCR_S', 'Z3_NO_UCR_N1', 'Pa_NO_BSR_SI71', 'ISA_NO_NPR_S', 'IP_NO_PLR_S', 'Pb_NO_PCR_V', 'ECI_NO_PCR_CV')}, 22: {'feature_idx': (0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 18, 19, 20, 22, 23, 25, 26, 27), 'cv_scores': array([0.15718419, 0.00118483, 0.16349047, 0.00118765, 0.16839135,
       0.00116009, 0.1619469 , 0.        , 0.07685739, 0.07538995]), 'avg_score': 0.0806792814710358, 'feature_names': ('IP_ES_25_N1', 'Z3_IB_4_N1', 'Z1_IB_10_N1', 'Z1_IB_5_N1', 'Z3_IB_8_N1', 'ECI_IB_4_N1', 'Gs(U)_IB_12_N1', 'Gs(U)_IB_68_N1', 'Gs(U)_IB_60_N1', 'Z1_NO_sideL35_M', 'Z1_NO_sideR35_CV', 'Pb_NO_sideR35_S', 'Z1_NO_PRT_CV', 'Z2_NO_AHR_CV', 'Gs(U)_NO_ALR_SI71', 'Z3_NO_UCR_S', 'Z3_NO_UCR_N1', 'Pa_NO_BSR_SI71', 'ISA_NO_NPR_S', 'IP_NO_PLR_S', 'Pb_NO_PCR_V', 'ECI_NO_PCR_CV')}, 21: {'feature_idx': (0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 18, 19, 20, 22, 23, 25, 26), 'cv_scores': array([0.15913371, 0.00117786, 0.16045845, 0.00120773, 0.16805171,
       0.00114811, 0.16328332, 0.        , 0.0780446 , 0.07745267]), 'avg_score': 0.08099581456667512, 'feature_names': ('IP_ES_25_N1', 'Z3_IB_4_N1', 'Z1_IB_10_N1', 'Z1_IB_5_N1', 'Z3_IB_8_N1', 'ECI_IB_4_N1', 'Gs(U)_IB_12_N1', 'Gs(U)_IB_68_N1', 'Gs(U)_IB_60_N1', 'Z1_NO_sideL35_M', 'Z1_NO_sideR35_CV', 'Pb_NO_sideR35_S', 'Z1_NO_PRT_CV', 'Z2_NO_AHR_CV', 'Gs(U)_NO_ALR_SI71', 'Z3_NO_UCR_S', 'Z3_NO_UCR_N1', 'Pa_NO_BSR_SI71', 'ISA_NO_NPR_S', 'IP_NO_PLR_S', 'Pb_NO_PCR_V')}, 20: {'feature_idx': (0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 18, 19, 22, 23, 25, 26), 'cv_scores': array([0.15659341, 0.0011655 , 0.1645933 , 0.        , 0.16977612,
       0.00113636, 0.16391941, 0.        , 0.07711651, 0.07225914]), 'avg_score': 0.08065597553581576, 'feature_names': ('IP_ES_25_N1', 'Z3_IB_4_N1', 'Z1_IB_10_N1', 'Z1_IB_5_N1', 'Z3_IB_8_N1', 'ECI_IB_4_N1', 'Gs(U)_IB_12_N1', 'Gs(U)_IB_68_N1', 'Gs(U)_IB_60_N1', 'Z1_NO_sideL35_M', 'Z1_NO_sideR35_CV', 'Pb_NO_sideR35_S', 'Z1_NO_PRT_CV', 'Z2_NO_AHR_CV', 'Gs(U)_NO_ALR_SI71', 'Z3_NO_UCR_S', 'Pa_NO_BSR_SI71', 'ISA_NO_NPR_S', 'IP_NO_PLR_S', 'Pb_NO_PCR_V')}, 19: {'feature_idx': (0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 13, 14, 17, 18, 19, 22, 23, 25, 26), 'cv_scores': array([0.15710254, 0.00118765, 0.16455696, 0.        , 0.16401125,
       0.00116144, 0.16482505, 0.        , 0.07395234, 0.07525952]), 'avg_score': 0.08020567405694701, 'feature_names': ('IP_ES_25_N1', 'Z3_IB_4_N1', 'Z1_IB_10_N1', 'Z1_IB_5_N1', 'Z3_IB_8_N1', 'ECI_IB_4_N1', 'Gs(U)_IB_12_N1', 'Gs(U)_IB_68_N1', 'Gs(U)_IB_60_N1', 'Z1_NO_sideL35_M', 'Z1_NO_sideR35_CV', 'Pb_NO_sideR35_S', 'Z2_NO_AHR_CV', 'Gs(U)_NO_ALR_SI71', 'Z3_NO_UCR_S', 'Pa_NO_BSR_SI71', 'ISA_NO_NPR_S', 'IP_NO_PLR_S', 'Pb_NO_PCR_V')}, 18: {'feature_idx': (0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 13, 14, 17, 18, 19, 23, 25, 26), 'cv_scores': array([0.15799615, 0.00117371, 0.16197866, 0.        , 0.16471647,
       0.0011655 , 0.16103203, 0.        , 0.07865169, 0.07811081]), 'avg_score': 0.08048250118810511, 'feature_names': ('IP_ES_25_N1', 'Z3_IB_4_N1', 'Z1_IB_10_N1', 'Z1_IB_5_N1', 'Z3_IB_8_N1', 'ECI_IB_4_N1', 'Gs(U)_IB_12_N1', 'Gs(U)_IB_68_N1', 'Gs(U)_IB_60_N1', 'Z1_NO_sideL35_M', 'Z1_NO_sideR35_CV', 'Pb_NO_sideR35_S', 'Z2_NO_AHR_CV', 'Gs(U)_NO_ALR_SI71', 'Z3_NO_UCR_S', 'ISA_NO_NPR_S', 'IP_NO_PLR_S', 'Pb_NO_PCR_V')}, 17: {'feature_idx': (0, 1, 2, 3, 4, 5, 7, 8, 10, 13, 14, 17, 18, 19, 23, 25, 26), 'cv_scores': array([0.15621986, 0.00116144, 0.16061185, 0.00122549, 0.16469518,
       0.00115473, 0.16651418, 0.        , 0.07534247, 0.07699038]), 'avg_score': 0.08039155850159059, 'feature_names': ('IP_ES_25_N1', 'Z3_IB_4_N1', 'Z1_IB_10_N1', 'Z1_IB_5_N1', 'Z3_IB_8_N1', 'ECI_IB_4_N1', 'Gs(U)_IB_12_N1', 'Gs(U)_IB_68_N1', 'Gs(U)_IB_60_N1', 'Z1_NO_sideR35_CV', 'Pb_NO_sideR35_S', 'Z2_NO_AHR_CV', 'Gs(U)_NO_ALR_SI71', 'Z3_NO_UCR_S', 'ISA_NO_NPR_S', 'IP_NO_PLR_S', 'Pb_NO_PCR_V')}, 16: {'feature_idx': (0, 1, 2, 3, 4, 5, 7, 8, 10, 13, 14, 18, 19, 23, 25, 26), 'cv_scores': array([0.15699334, 0.00116959, 0.16045845, 0.        , 0.16776007,
       0.00118203, 0.16088889, 0.        , 0.07831325, 0.07555178]), 'avg_score': 0.08023174156987828, 'feature_names': ('IP_ES_25_N1', 'Z3_IB_4_N1', 'Z1_IB_10_N1', 'Z1_IB_5_N1', 'Z3_IB_8_N1', 'ECI_IB_4_N1', 'Gs(U)_IB_12_N1', 'Gs(U)_IB_68_N1', 'Gs(U)_IB_60_N1', 'Z1_NO_sideR35_CV', 'Pb_NO_sideR35_S', 'Gs(U)_NO_ALR_SI71', 'Z3_NO_UCR_S', 'ISA_NO_NPR_S', 'IP_NO_PLR_S', 'Pb_NO_PCR_V')}, 15: {'feature_idx': (0, 1, 2, 3, 4, 5, 7, 8, 10, 13, 18, 19, 23, 25, 26), 'cv_scores': array([0.16059113, 0.00118765, 0.15829384, 0.        , 0.16604824,
       0.00114155, 0.16727273, 0.        , 0.08098592, 0.07558645]), 'avg_score': 0.08111074996456533, 'feature_names': ('IP_ES_25_N1', 'Z3_IB_4_N1', 'Z1_IB_10_N1', 'Z1_IB_5_N1', 'Z3_IB_8_N1', 'ECI_IB_4_N1', 'Gs(U)_IB_12_N1', 'Gs(U)_IB_68_N1', 'Gs(U)_IB_60_N1', 'Z1_NO_sideR35_CV', 'Gs(U)_NO_ALR_SI71', 'Z3_NO_UCR_S', 'ISA_NO_NPR_S', 'IP_NO_PLR_S', 'Pb_NO_PCR_V')}, 14: {'feature_idx': (0, 1, 2, 3, 4, 5, 7, 8, 10, 13, 19, 23, 25, 26), 'cv_scores': array([0.16018957, 0.00115075, 0.1615087 , 0.        , 0.16361974,
       0.00114155, 0.16091954, 0.        , 0.07888041, 0.07604895]), 'avg_score': 0.08034592204812366, 'feature_names': ('IP_ES_25_N1', 'Z3_IB_4_N1', 'Z1_IB_10_N1', 'Z1_IB_5_N1', 'Z3_IB_8_N1', 'ECI_IB_4_N1', 'Gs(U)_IB_12_N1', 'Gs(U)_IB_68_N1', 'Gs(U)_IB_60_N1', 'Z1_NO_sideR35_CV', 'Z3_NO_UCR_S', 'ISA_NO_NPR_S', 'IP_NO_PLR_S', 'Pb_NO_PCR_V')}, 13: {'feature_idx': (1, 2, 3, 4, 5, 7, 8, 10, 13, 19, 23, 25, 26), 'cv_scores': array([0.15469613, 0.0011655 , 0.1610338 , 0.        , 0.16213768,
       0.00113122, 0.15957447, 0.        , 0.07863248, 0.07555178]), 'avg_score': 0.07939230632578612, 'feature_names': ('Z3_IB_4_N1', 'Z1_IB_10_N1', 'Z1_IB_5_N1', 'Z3_IB_8_N1', 'ECI_IB_4_N1', 'Gs(U)_IB_12_N1', 'Gs(U)_IB_68_N1', 'Gs(U)_IB_60_N1', 'Z1_NO_sideR35_CV', 'Z3_NO_UCR_S', 'ISA_NO_NPR_S', 'IP_NO_PLR_S', 'Pb_NO_PCR_V')}, 12: {'feature_idx': (1, 2, 3, 4, 5, 7, 8, 10, 19, 23, 25, 26), 'cv_scores': array([0.15251142, 0.00113895, 0.15678776, 0.        , 0.16245487,
       0.00110988, 0.16143106, 0.        , 0.07969152, 0.07632264]), 'avg_score': 0.07914481000371829, 'feature_names': ('Z3_IB_4_N1', 'Z1_IB_10_N1', 'Z1_IB_5_N1', 'Z3_IB_8_N1', 'ECI_IB_4_N1', 'Gs(U)_IB_12_N1', 'Gs(U)_IB_68_N1', 'Gs(U)_IB_60_N1', 'Z3_NO_UCR_S', 'ISA_NO_NPR_S', 'IP_NO_PLR_S', 'Pb_NO_PCR_V')}, 11: {'feature_idx': (1, 2, 3, 4, 5, 7, 8, 10, 23, 25, 26), 'cv_scores': array([0.15237226, 0.00116822, 0.15604801, 0.        , 0.16126126,
       0.00113122, 0.16368515, 0.        , 0.07874682, 0.07639485]), 'avg_score': 0.07908078113947452, 'feature_names': ('Z3_IB_4_N1', 'Z1_IB_10_N1', 'Z1_IB_5_N1', 'Z3_IB_8_N1', 'ECI_IB_4_N1', 'Gs(U)_IB_12_N1', 'Gs(U)_IB_68_N1', 'Gs(U)_IB_60_N1', 'ISA_NO_NPR_S', 'IP_NO_PLR_S', 'Pb_NO_PCR_V')}, 10: {'feature_idx': (1, 2, 4, 5, 7, 8, 10, 23, 25, 26), 'cv_scores': array([0.15433213, 0.00112613, 0.15483871, 0.        , 0.16470588,
       0.00110742, 0.16358839, 0.        , 0.07717842, 0.07478992]), 'avg_score': 0.07916669975366768, 'feature_names': ('Z3_IB_4_N1', 'Z1_IB_10_N1', 'Z3_IB_8_N1', 'ECI_IB_4_N1', 'Gs(U)_IB_12_N1', 'Gs(U)_IB_68_N1', 'Gs(U)_IB_60_N1', 'ISA_NO_NPR_S', 'IP_NO_PLR_S', 'Pb_NO_PCR_V')}, 9: {'feature_idx': (1, 2, 4, 5, 7, 8, 10, 23, 25), 'cv_scores': array([0.15205725, 0.00112613, 0.15610652, 0.        , 0.16021127,
       0.00110742, 0.16049383, 0.        , 0.07597027, 0.07627119]), 'avg_score': 0.07833438643704513, 'feature_names': ('Z3_IB_4_N1', 'Z1_IB_10_N1', 'Z3_IB_8_N1', 'ECI_IB_4_N1', 'Gs(U)_IB_12_N1', 'Gs(U)_IB_68_N1', 'Gs(U)_IB_60_N1', 'ISA_NO_NPR_S', 'IP_NO_PLR_S')}, 8: {'feature_idx': (1, 4, 5, 7, 8, 10, 23, 25), 'cv_scores': array([0.1497373 , 0.00110497, 0.15523466, 0.        , 0.16143498,
       0.00107991, 0.16134599, 0.        , 0.07781457, 0.0762987 ]), 'avg_score': 0.07840510823337679, 'feature_names': ('Z3_IB_4_N1', 'Z3_IB_8_N1', 'ECI_IB_4_N1', 'Gs(U)_IB_12_N1', 'Gs(U)_IB_68_N1', 'Gs(U)_IB_60_N1', 'ISA_NO_NPR_S', 'IP_NO_PLR_S')}, 7: {'feature_idx': (1, 4, 7, 8, 10, 23, 25), 'cv_scores': array([0.15173026, 0.00111982, 0.1569873 , 0.        , 0.15748031,
       0.00110011, 0.16071429, 0.        , 0.07513989, 0.07380373]), 'avg_score': 0.07780757034882406, 'feature_names': ('Z3_IB_4_N1', 'Z3_IB_8_N1', 'Gs(U)_IB_12_N1', 'Gs(U)_IB_68_N1', 'Gs(U)_IB_60_N1', 'ISA_NO_NPR_S', 'IP_NO_PLR_S')}, 6: {'feature_idx': (1, 4, 7, 8, 23, 25), 'cv_scores': array([0.14762742, 0.00111483, 0.15314494, 0.        , 0.16010499,
       0.00110011, 0.15856777, 0.        , 0.07797428, 0.07481899]), 'avg_score': 0.0774453319143933, 'feature_names': ('Z3_IB_4_N1', 'Z3_IB_8_N1', 'Gs(U)_IB_12_N1', 'Gs(U)_IB_68_N1', 'ISA_NO_NPR_S', 'IP_NO_PLR_S')}, 5: {'feature_idx': (1, 4, 8, 23, 25), 'cv_scores': array([0.15057573, 0.00111235, 0.15138023, 0.        , 0.15443038,
       0.00106496, 0.15609349, 0.        , 0.07717042, 0.0752    ]), 'avg_score': 0.07670275589383031, 'feature_names': ('Z3_IB_4_N1', 'Z3_IB_8_N1', 'Gs(U)_IB_68_N1', 'ISA_NO_NPR_S', 'IP_NO_PLR_S')}, 4: {'feature_idx': (1, 8, 23, 25), 'cv_scores': array([0.15081405, 0.00107066, 0.14964789, 0.        , 0.15521628,
       0.00105485, 0.1560166 , 0.        , 0.07847708, 0.07517619]), 'avg_score': 0.07674736117637573, 'feature_names': ('Z3_IB_4_N1', 'Gs(U)_IB_68_N1', 'ISA_NO_NPR_S', 'IP_NO_PLR_S')}, 3: {'feature_idx': (8, 23, 25), 'cv_scores': array([0.15141431, 0.00104932, 0.14995641, 0.00102145, 0.15124698,
       0.00100503, 0.15453074, 0.        , 0.07354056, 0.07214121]), 'avg_score': 0.0755906012093118, 'feature_names': ('Gs(U)_IB_68_N1', 'ISA_NO_NPR_S', 'IP_NO_PLR_S')}, 2: {'feature_idx': (23, 25), 'cv_scores': array([0.14576547, 0.0010352 , 0.14811715, 0.00102354, 0.14768264,
       0.00100402, 0.14912281, 0.        , 0.0741018 , 0.07148289]), 'avg_score': 0.07393355139224522, 'feature_names': ('ISA_NO_NPR_S', 'IP_NO_PLR_S')}, 1: {'feature_idx': (23,), 'cv_scores': array([0.13187773, 0.        , 0.14174757, 0.00179856, 0.15230635,
       0.00090171, 0.13548951, 0.        , 0.08265213, 0.07889734]), 'avg_score': 0.07256709131459835, 'feature_names': ('ISA_NO_NPR_S',)}}

#--------------------------------------------------------------------------------
class BayesianPNN(algorithms.PNN):
    # ['N', 'P']
    priors = np.array([1, 1])
    def __init__(self, class_priors=None, *pargs, **kwargs):
        kwargs['batch_size'] = None
        super(BayesianPNN, self).__init__(*pargs, **kwargs)
        if class_priors:
            BayesianPNN.priors = np.array(class_priors)
            self.priors = BayesianPNN.priors

    def predict_raw(self, X):
        raw_output = super(BayesianPNN, self).predict_raw(X)
        self.classes_ = self.classes

        # Bayesian Approach to prevent over prediction of minority class.
        return (raw_output.T * self.priors).T

#--------------------------------------------------------------------------------
def import_csv(file_path):
    return pd.read_csv(file_path)

#--------------------------------------------------------------------------------
def concatenate_frames(frames):
    return pd.concat(frames)

#--------------------------------------------------------------------------------
def _create_model_instance(model):
    normalizer = Normalizer()
    
    return Pipeline([('standardizer', StandardScaler()),
                     ('undersample', RandomUnderSampler(sampling_strategy=RATIO)),
#                      ('smote', SMOTE()),
                     ('normalizer', normalizer),
                     ('pnn', model)])

#--------------------------------------------------------------------------------
def create_model_instance(class_priors=None, std=10):
    # Creates an instance of the model with the given standard deviation and
    # weighs the score of each class with the given priors. The model performs
    # SMOTE before fitting training data.
    if class_priors == None:
        class_priors = [1, 1]

    model = BayesianPNN(class_priors, std=std)

    return _create_model_instance(model)

#--------------------------------------------------------------------------------
def create_ensemble_model_instance(class_priors=None, std=10):
    # Creates an instance of the model with the given standard deviation and
    # weighs the score of each class with the given priors. The model performs
    # SMOTE before fitting training data.
    if class_priors == None:
        class_priors = [1, 1]

    model = BayesianPNN(class_priors, std=std)
    model = BaggingClassifier(base_estimator=model, n_estimators=10, n_jobs=-1)

    return _create_model_instance(model)

#--------------------------------------------------------------------------------
def remove_outliers_iqr(df):
    # Remove all outliers outside the 95% confidence interval.
    x = data.drop('class', 1)

    Q1 = x.quantile(0.25)
    Q3 = x.quantile(0.75)
    IQR = Q3 - Q1

    return df[~((x < (Q1 - 1.5 * IQR)) |
                (x > (Q3 + 1.5 * IQR))).any(axis=1)]

#--------------------------------------------------------------------------------
def remove_outliers_normal(df):
    # Remove all outliers outside the 95% confidence interval.
    x = data.drop('class', 1)
    
    return df[(np.abs(stats.zscore(x)) < 3).all(axis=1)]

#--------------------------------------------------------------------------------
def backward_elimination(model, X, y):
    # Return the optimal set of features to use for classification. Uses 10 fold
    # cross-validation to evaluate the performance of each feature set. Selects
    # feature set that achieves maximal precision.
    sbs = Sfs(model,
              k_features='parsimonious',
              forward=False,
              floating=False,
              cv=10,
              scoring=make_scorer(precision_score, greater_is_better=True, 
                                  needs_proba=False, pos_label='P'),
              n_jobs=-1,
              verbose=2)
    sbs.fit(X, y)
    print(80*'-')
    print(sbs.subsets_)
    print(80*'-')
    return sbs.k_feature_names_

#--------------------------------------------------------------------------------
def feature_extraction(X, y):
    # Return the optimal set of features to use for classification. Uses 10 fold
    # cross-validation to evaluate the performance of each feature set. Selects
    # feature set that achieves maximal precision.
    lda = PrincipalComponentAnalysis(n_discriminants=20)
    y = y.replace('N', 0).replace('P', 1)
    lda.fit(np.array(X), y)
    X_lda = lda.transform(X)
    return X_lda

#--------------------------------------------------------------------------------
def evaluate_model(model, X, y, cv=10, std=10, final=''):
    # Returns the precision at 50% recall.
    y_pred_prob = cross_val_predict(model, X, y, cv=cv, method='predict_proba')

    # Plot the Precision-Recall curve.
    pr, re, thresholds = precision_recall_curve(y, y_pred_prob[:, 1],
                                                pos_label=POSITIVE_LABEL)

    indices_above_50_re = np.where(re >= 0.5)

    pr_beyond_50 = pr[indices_above_50_re]
    thr_beyond_50 = thresholds[indices_above_50_re]
    re_beyond_50 = re[indices_above_50_re]
    
    max_pr = pr_beyond_50[np.argmax(pr_beyond_50)]
    threshold_at_max_pr = thr_beyond_50[np.argmax(pr_beyond_50)]
    re_at_max_pr = re_beyond_50[np.argmax(pr_beyond_50)]

    # Get the predicted classes of the model.
    classes = ['N', 'P']
    y_pred = np.array(['P' if y >= threshold_at_max_pr else 'N'
                       for y in y_pred_prob[:, 1]])

    # Get the confusion matrix: tn, fp, tp, fp = conf_matrix.ravel().
    conf_matrix = confusion_matrix(y, y_pred, labels=classes)
    tn, fp, fn, tp = conf_matrix.ravel()

    # Calculate precision's standard deviation.
    pr_std = get_precision_std(tp, tp + fp)
    
    print('-' * 30 + 'std_param = {}'.format(std) + 30 * '-')

    print('TP {}, FP {}'.format(tp, fp))
    print('FN {}, TN {}'.format(fn, tn))

    print('Threshold: {}'.format(threshold_at_max_pr))
    print('Recall: {}'.format(re_at_max_pr))
    print('Precision: {}'.format(max_pr))
    print('Standard deviation: {}'.format(pr_std))

    # Compute accuracy (for meta-learning).
    acc_score = accuracy_score(y, y_pred)
    print("Accuracy : {}".format(acc_score))

    plt.figure()
    plt.step(re, pr, color='b', alpha=0.8, where='post')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve for {} Features with {} std'
              .format(X.shape[1], std))
    plt.savefig(os.path.join(BASE_DIR, 'Pr_Re_{}_features_{}_std_{}.jpg'.format(
        X.shape[1], std, final)))

    print('-' * 80)
    return max_pr, pr_std

#--------------------------------------------------------------------------------
def get_precision_std(tp, n):
    p = float(tp) / n
    variance_of_sum = p * (1 - p) / n
    std = variance_of_sum ** 0.5
    return std

#--------------------------------------------------------------------------------
if __name__ == "__main__":
    csv_path_training = os.path.join(BASE_DIR, CSV_FILE_TRAINING)
    csv_path_calibration = os.path.join(BASE_DIR, CSV_FILE_CALIBRATION)

    # All of our data collected.
    data = concatenate_frames([import_csv(csv_path_training),
                               import_csv(csv_path_calibration)])
    
    # Drop id column from data.
    data = data.drop(['id'], 1)

    # Segregate the negative and positive populations.
    negative_data = data.loc[data['class'] == 'N']
    positive_data = data.loc[data['class'] == 'P']

    # Calculate priors.
    negative_prior = len(negative_data.index) / (1. * len(data.index))
    positive_prior = len(positive_data.index) / (1. * len(data.index))
    priors = [negative_prior, positive_prior]
    priors = [1, 1]
    
    new_data = data
    print('Number of outliers removed: {}'.format(abs(len(new_data.index) -
                                                  len(data.index))))
    data = new_data
    
    # Separate features from their corresponding class.
    X = data.drop(['class'], 1)
    y = data['class']

    # Instantiate PNN model with std. dev. 1.
#     model = create_model_instance(priors, 2.8)
#     evaluate_model(model, X, y, std=2.8)
    #----------------------------------------------------------------------
    # Select features using backward elimination. 10 fold cv with minimum 12
    # features and maximum 28 features.
#     selected_features = backward_elimination(model, X, y)
    selected_features = ALL_SUBSETS_47[14]['feature_names']
    selected_features = list(selected_features)
    print('Features selected: {}'.format(selected_features))

    X_new = X[selected_features]

    print('Number of features selected: {}'.format(len(X_new.columns)))

    # Tuning classifier based on maximum precision achieved
    # Use the Gaussian Naive Bayes classifier.
    # Trial all standard deviations and pick the standard deviation
    # that results in the maximum precision.
    max_pr = 0
    max_pr_std = 0
    max_std_param = 0
    STD_PARAMS = np.arange(0.2, 10, 0.2)
    for std_param in STD_PARAMS:
        model = create_model_instance(priors, std_param)
        pr_at_re50, pr_std = evaluate_model(model, X_new, y, std=std_param)
        if pr_at_re50 >= max_pr:
            max_pr = pr_at_re50
            max_pr_std = pr_std
            max_std_param = std_param

    print('Optimal standard deviation is {}.'.format(max_std_param))
    print('Final Pr@Re 50 achieved is {} +/- {}'.format(
        max_pr,
        max_pr_std))
    

    # Final model without meta-learning.
    model = create_model_instance(priors, max_std_param)
    pr_at_re50, pr_std = evaluate_model(model, X_new, y, std=max_std_param,
                                        cv=LeaveOneOut(), final='FINAL')
    
    print('Final max Pr@Re50% achieved is {} +/- {}'.format(
        pr_at_re50,
        pr_std))

    # Meta-learning model.
    model = create_ensemble_model_instance(priors, max_std_param)
    pr_at_re50, pr_std = evaluate_model(model, X_new, y, std=max_std_param,
                                        cv=10, final='FINAL_META_10')

    print('Final max Pr@Re50% with meta-learning achieved is {} +/- {}'.format(
        pr_at_re50,
        pr_std))


# In[14]:


#--------------------------------------------------------------------------------
#
# Project implementation.
#
#--------------------------------------------------------------------------------
# !pip install neupy
# !pip install --upgrade mlxtend
import pandas as pd
from scipy import stats
import numpy as np
from numpy import array
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_predict, StratifiedKFold,     LeaveOneOut

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score,     recall_score, roc_curve, precision_recall_curve, make_scorer

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.feature_selection import SelectFromModel
from neupy import algorithms
from neupy.utils import format_data, iters
from mlxtend.feature_selection import SequentialFeatureSelector as Sfs
from mlxtend.feature_extraction import LinearDiscriminantAnalysis,     PrincipalComponentAnalysis
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import BaggingClassifier
import os

#--------------------------------------------------------------------------------
np.random.seed(0)
# from google.colab import drive
# drive.mount('/content/gdrive')

#--------------------------------------------------------------------------------

RATIO = 4/7.
BASE_DIR = F"C:/Users/zalat/Downloads/project/"
CSV_FILE_TRAINING = F"csv_result-Descriptors_Training.csv"
CSV_FILE_CALIBRATION = F"csv_result-Descriptors_Calibration.csv"
CSV_FILE_BLIND = F"Blind_Test_features.csv"

POSITIVE_LABEL = 'P'
NEGATIVE_LABEL = 'N'
SELECTED_FEATURES = ('IP_ES_25_N1', 'Z1_IB_10_N1', 'Gs(U)_IB_12_N1',
                     'Gs(U)_IB_68_N1', 'Gs(U)_IB_58_N1', 'Gs(U)_IB_60_N1',
                     'Z1_NO_sideL35_M', 'HP_NO_sideL35_CV',
                     'Z1_NO_sideR35_CV', 'Pb_NO_sideR35_S',
                     'IP_NO_sideL35_SI71', 'Z1_NO_PRT_CV', 'Z2_NO_AHR_CV',
                     'Gs(U)_NO_ALR_SI71', 'Z3_NO_UCR_S', 'Z3_NO_UCR_N1',
                     'ECI_NO_UCR_CV', 'Pa_NO_BSR_SI71', 'ISA_NO_NPR_S',
                     'Z3_NO_NPR_V', 'IP_NO_PLR_S', 'Pb_NO_PCR_V',
                     'ECI_NO_PCR_CV')
SELECTED_FEATURES_5 = ('IP_NO_PLR_S', 'ISA_NO_NPR_S', 'IP_ES_25_N1',
                       'Gs(U)_NO_ALR_SI71', 'Z1_NO_PRT_CV', 'Gs(U)_IB_58_N1',
                       'Z1_IB_5_N1', 'Gs(U)_IB_68_N1', 'Gs(U)_IB_60_N1')
# Selected using most discriminating features and LSVC.
SELECTED_FEATURES_5 = ('IP_NO_PLR_S', 'ISA_NO_NPR_S', 'IP_ES_25_N1',
                       'Gs(U)_NO_ALR_SI71', 'Z1_NO_PRT_CV', 'Gs(U)_IB_58_N1',
                       'Z1_IB_5_N1', 'Gs(U)_IB_68_N1', 'Gs(U)_IB_60_N1',
                       'Z3_NO_UCR_S', 'Z2_NO_AHR_CV')
# Selected using only infogain.
SELECTED_FEATURES_5 = ('IP_NO_PLR_S', 'ISA_NO_NPR_S', 'Z1_NO_PRT_CV',
                       'Z2_NO_AHR_CV', 'Gs(U)_NO_ALR_SI71', 'HP_NO_sideL35_CV',
                       'Z1_NO_sideR35_CV', 'Z3_IB_4_N1', 'Z1_IB_5_N1',
                       'Z1_NO_sideL35_M', 'Pb_NO_sideR35_S')
# Selected using backward elimination. (1st attempt)
SELECTED_FEATURES_6 = ('Z1_IB_10_N1', 'Z1_IB_5_N1')

# Improved backward el. (2nd attempt)
SELECTED_FEATURES_7 = ('Z1_IB_10_N1', 'Z1_IB_5_N1', 'ECI_NO_UCR_CV')

ALL_SUBSETS_47 = {28: {'feature_idx': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27), 'cv_scores': array([0.15897436, 0.00118343, 0.15686275, 0.00121507, 0.15996503,
       0.00121507, 0.16619183, 0.        , 0.07858377, 0.07067425]), 'avg_score': 0.07948655511966006, 'feature_names': ('IP_ES_25_N1', 'Z3_IB_4_N1', 'Z1_IB_10_N1', 'Z1_IB_5_N1', 'Z3_IB_8_N1', 'ECI_IB_4_N1', 'ECI_IB_5_N1', 'Gs(U)_IB_12_N1', 'Gs(U)_IB_68_N1', 'Gs(U)_IB_58_N1', 'Gs(U)_IB_60_N1', 'Z1_NO_sideL35_M', 'HP_NO_sideL35_CV', 'Z1_NO_sideR35_CV', 'Pb_NO_sideR35_S', 'IP_NO_sideL35_SI71', 'Z1_NO_PRT_CV', 'Z2_NO_AHR_CV', 'Gs(U)_NO_ALR_SI71', 'Z3_NO_UCR_S', 'Z3_NO_UCR_N1', 'ECI_NO_UCR_CV', 'Pa_NO_BSR_SI71', 'ISA_NO_NPR_S', 'Z3_NO_NPR_V', 'IP_NO_PLR_S', 'Pb_NO_PCR_V', 'ECI_NO_PCR_CV')}, 27: {'feature_idx': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27), 'cv_scores': array([0.15646941, 0.00116009, 0.16135458, 0.        , 0.16332117,
       0.00116009, 0.16233766, 0.        , 0.08268734, 0.07791096]), 'avg_score': 0.08064013031391057, 'feature_names': ('IP_ES_25_N1', 'Z3_IB_4_N1', 'Z1_IB_10_N1', 'Z1_IB_5_N1', 'Z3_IB_8_N1', 'ECI_IB_4_N1', 'ECI_IB_5_N1', 'Gs(U)_IB_12_N1', 'Gs(U)_IB_68_N1', 'Gs(U)_IB_58_N1', 'Gs(U)_IB_60_N1', 'Z1_NO_sideL35_M', 'HP_NO_sideL35_CV', 'Z1_NO_sideR35_CV', 'Pb_NO_sideR35_S', 'IP_NO_sideL35_SI71', 'Z1_NO_PRT_CV', 'Z2_NO_AHR_CV', 'Gs(U)_NO_ALR_SI71', 'Z3_NO_UCR_S', 'Z3_NO_UCR_N1', 'Pa_NO_BSR_SI71', 'ISA_NO_NPR_S', 'Z3_NO_NPR_V', 'IP_NO_PLR_S', 'Pb_NO_PCR_V', 'ECI_NO_PCR_CV')}, 26: {'feature_idx': (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27), 'cv_scores': array([0.16231884, 0.00115607, 0.161     , 0.        , 0.16263941,
       0.00117096, 0.16131989, 0.        , 0.07887818, 0.07876106]), 'avg_score': 0.08072444043294415, 'feature_names': ('IP_ES_25_N1', 'Z3_IB_4_N1', 'Z1_IB_10_N1', 'Z1_IB_5_N1', 'Z3_IB_8_N1', 'ECI_IB_4_N1', 'Gs(U)_IB_12_N1', 'Gs(U)_IB_68_N1', 'Gs(U)_IB_58_N1', 'Gs(U)_IB_60_N1', 'Z1_NO_sideL35_M', 'HP_NO_sideL35_CV', 'Z1_NO_sideR35_CV', 'Pb_NO_sideR35_S', 'IP_NO_sideL35_SI71', 'Z1_NO_PRT_CV', 'Z2_NO_AHR_CV', 'Gs(U)_NO_ALR_SI71', 'Z3_NO_UCR_S', 'Z3_NO_UCR_N1', 'Pa_NO_BSR_SI71', 'ISA_NO_NPR_S', 'Z3_NO_NPR_V', 'IP_NO_PLR_S', 'Pb_NO_PCR_V', 'ECI_NO_PCR_CV')}, 25: {'feature_idx': (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 25, 26, 27), 'cv_scores': array([0.15819751, 0.00118624, 0.16129032, 0.        , 0.16591928,
       0.00118064, 0.16074766, 0.        , 0.08093525, 0.07733333]), 'avg_score': 0.08067902381306255, 'feature_names': ('IP_ES_25_N1', 'Z3_IB_4_N1', 'Z1_IB_10_N1', 'Z1_IB_5_N1', 'Z3_IB_8_N1', 'ECI_IB_4_N1', 'Gs(U)_IB_12_N1', 'Gs(U)_IB_68_N1', 'Gs(U)_IB_58_N1', 'Gs(U)_IB_60_N1', 'Z1_NO_sideL35_M', 'HP_NO_sideL35_CV', 'Z1_NO_sideR35_CV', 'Pb_NO_sideR35_S', 'IP_NO_sideL35_SI71', 'Z1_NO_PRT_CV', 'Z2_NO_AHR_CV', 'Gs(U)_NO_ALR_SI71', 'Z3_NO_UCR_S', 'Z3_NO_UCR_N1', 'Pa_NO_BSR_SI71', 'ISA_NO_NPR_S', 'IP_NO_PLR_S', 'Pb_NO_PCR_V', 'ECI_NO_PCR_CV')}, 24: {'feature_idx': (0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 25, 26, 27), 'cv_scores': array([0.15824595, 0.00113895, 0.16079924, 0.        , 0.16849817,
       0.00115473, 0.16501353, 0.        , 0.07926829, 0.07574468]), 'avg_score': 0.08098635416486549, 'feature_names': ('IP_ES_25_N1', 'Z3_IB_4_N1', 'Z1_IB_10_N1', 'Z1_IB_5_N1', 'Z3_IB_8_N1', 'ECI_IB_4_N1', 'Gs(U)_IB_12_N1', 'Gs(U)_IB_68_N1', 'Gs(U)_IB_60_N1', 'Z1_NO_sideL35_M', 'HP_NO_sideL35_CV', 'Z1_NO_sideR35_CV', 'Pb_NO_sideR35_S', 'IP_NO_sideL35_SI71', 'Z1_NO_PRT_CV', 'Z2_NO_AHR_CV', 'Gs(U)_NO_ALR_SI71', 'Z3_NO_UCR_S', 'Z3_NO_UCR_N1', 'Pa_NO_BSR_SI71', 'ISA_NO_NPR_S', 'IP_NO_PLR_S', 'Pb_NO_PCR_V', 'ECI_NO_PCR_CV')}, 23: {'feature_idx': (0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 22, 23, 25, 26, 27), 'cv_scores': array([0.15727273, 0.00117096, 0.15950334, 0.0011976 , 0.16872038,
       0.00116414, 0.16233184, 0.        , 0.07865169, 0.07491582]), 'avg_score': 0.08049285075098568, 'feature_names': ('IP_ES_25_N1', 'Z3_IB_4_N1', 'Z1_IB_10_N1', 'Z1_IB_5_N1', 'Z3_IB_8_N1', 'ECI_IB_4_N1', 'Gs(U)_IB_12_N1', 'Gs(U)_IB_68_N1', 'Gs(U)_IB_60_N1', 'Z1_NO_sideL35_M', 'HP_NO_sideL35_CV', 'Z1_NO_sideR35_CV', 'Pb_NO_sideR35_S', 'Z1_NO_PRT_CV', 'Z2_NO_AHR_CV', 'Gs(U)_NO_ALR_SI71', 'Z3_NO_UCR_S', 'Z3_NO_UCR_N1', 'Pa_NO_BSR_SI71', 'ISA_NO_NPR_S', 'IP_NO_PLR_S', 'Pb_NO_PCR_V', 'ECI_NO_PCR_CV')}, 22: {'feature_idx': (0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 18, 19, 20, 22, 23, 25, 26, 27), 'cv_scores': array([0.15718419, 0.00118483, 0.16349047, 0.00118765, 0.16839135,
       0.00116009, 0.1619469 , 0.        , 0.07685739, 0.07538995]), 'avg_score': 0.0806792814710358, 'feature_names': ('IP_ES_25_N1', 'Z3_IB_4_N1', 'Z1_IB_10_N1', 'Z1_IB_5_N1', 'Z3_IB_8_N1', 'ECI_IB_4_N1', 'Gs(U)_IB_12_N1', 'Gs(U)_IB_68_N1', 'Gs(U)_IB_60_N1', 'Z1_NO_sideL35_M', 'Z1_NO_sideR35_CV', 'Pb_NO_sideR35_S', 'Z1_NO_PRT_CV', 'Z2_NO_AHR_CV', 'Gs(U)_NO_ALR_SI71', 'Z3_NO_UCR_S', 'Z3_NO_UCR_N1', 'Pa_NO_BSR_SI71', 'ISA_NO_NPR_S', 'IP_NO_PLR_S', 'Pb_NO_PCR_V', 'ECI_NO_PCR_CV')}, 21: {'feature_idx': (0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 18, 19, 20, 22, 23, 25, 26), 'cv_scores': array([0.15913371, 0.00117786, 0.16045845, 0.00120773, 0.16805171,
       0.00114811, 0.16328332, 0.        , 0.0780446 , 0.07745267]), 'avg_score': 0.08099581456667512, 'feature_names': ('IP_ES_25_N1', 'Z3_IB_4_N1', 'Z1_IB_10_N1', 'Z1_IB_5_N1', 'Z3_IB_8_N1', 'ECI_IB_4_N1', 'Gs(U)_IB_12_N1', 'Gs(U)_IB_68_N1', 'Gs(U)_IB_60_N1', 'Z1_NO_sideL35_M', 'Z1_NO_sideR35_CV', 'Pb_NO_sideR35_S', 'Z1_NO_PRT_CV', 'Z2_NO_AHR_CV', 'Gs(U)_NO_ALR_SI71', 'Z3_NO_UCR_S', 'Z3_NO_UCR_N1', 'Pa_NO_BSR_SI71', 'ISA_NO_NPR_S', 'IP_NO_PLR_S', 'Pb_NO_PCR_V')}, 20: {'feature_idx': (0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 18, 19, 22, 23, 25, 26), 'cv_scores': array([0.15659341, 0.0011655 , 0.1645933 , 0.        , 0.16977612,
       0.00113636, 0.16391941, 0.        , 0.07711651, 0.07225914]), 'avg_score': 0.08065597553581576, 'feature_names': ('IP_ES_25_N1', 'Z3_IB_4_N1', 'Z1_IB_10_N1', 'Z1_IB_5_N1', 'Z3_IB_8_N1', 'ECI_IB_4_N1', 'Gs(U)_IB_12_N1', 'Gs(U)_IB_68_N1', 'Gs(U)_IB_60_N1', 'Z1_NO_sideL35_M', 'Z1_NO_sideR35_CV', 'Pb_NO_sideR35_S', 'Z1_NO_PRT_CV', 'Z2_NO_AHR_CV', 'Gs(U)_NO_ALR_SI71', 'Z3_NO_UCR_S', 'Pa_NO_BSR_SI71', 'ISA_NO_NPR_S', 'IP_NO_PLR_S', 'Pb_NO_PCR_V')}, 19: {'feature_idx': (0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 13, 14, 17, 18, 19, 22, 23, 25, 26), 'cv_scores': array([0.15710254, 0.00118765, 0.16455696, 0.        , 0.16401125,
       0.00116144, 0.16482505, 0.        , 0.07395234, 0.07525952]), 'avg_score': 0.08020567405694701, 'feature_names': ('IP_ES_25_N1', 'Z3_IB_4_N1', 'Z1_IB_10_N1', 'Z1_IB_5_N1', 'Z3_IB_8_N1', 'ECI_IB_4_N1', 'Gs(U)_IB_12_N1', 'Gs(U)_IB_68_N1', 'Gs(U)_IB_60_N1', 'Z1_NO_sideL35_M', 'Z1_NO_sideR35_CV', 'Pb_NO_sideR35_S', 'Z2_NO_AHR_CV', 'Gs(U)_NO_ALR_SI71', 'Z3_NO_UCR_S', 'Pa_NO_BSR_SI71', 'ISA_NO_NPR_S', 'IP_NO_PLR_S', 'Pb_NO_PCR_V')}, 18: {'feature_idx': (0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 13, 14, 17, 18, 19, 23, 25, 26), 'cv_scores': array([0.15799615, 0.00117371, 0.16197866, 0.        , 0.16471647,
       0.0011655 , 0.16103203, 0.        , 0.07865169, 0.07811081]), 'avg_score': 0.08048250118810511, 'feature_names': ('IP_ES_25_N1', 'Z3_IB_4_N1', 'Z1_IB_10_N1', 'Z1_IB_5_N1', 'Z3_IB_8_N1', 'ECI_IB_4_N1', 'Gs(U)_IB_12_N1', 'Gs(U)_IB_68_N1', 'Gs(U)_IB_60_N1', 'Z1_NO_sideL35_M', 'Z1_NO_sideR35_CV', 'Pb_NO_sideR35_S', 'Z2_NO_AHR_CV', 'Gs(U)_NO_ALR_SI71', 'Z3_NO_UCR_S', 'ISA_NO_NPR_S', 'IP_NO_PLR_S', 'Pb_NO_PCR_V')}, 17: {'feature_idx': (0, 1, 2, 3, 4, 5, 7, 8, 10, 13, 14, 17, 18, 19, 23, 25, 26), 'cv_scores': array([0.15621986, 0.00116144, 0.16061185, 0.00122549, 0.16469518,
       0.00115473, 0.16651418, 0.        , 0.07534247, 0.07699038]), 'avg_score': 0.08039155850159059, 'feature_names': ('IP_ES_25_N1', 'Z3_IB_4_N1', 'Z1_IB_10_N1', 'Z1_IB_5_N1', 'Z3_IB_8_N1', 'ECI_IB_4_N1', 'Gs(U)_IB_12_N1', 'Gs(U)_IB_68_N1', 'Gs(U)_IB_60_N1', 'Z1_NO_sideR35_CV', 'Pb_NO_sideR35_S', 'Z2_NO_AHR_CV', 'Gs(U)_NO_ALR_SI71', 'Z3_NO_UCR_S', 'ISA_NO_NPR_S', 'IP_NO_PLR_S', 'Pb_NO_PCR_V')}, 16: {'feature_idx': (0, 1, 2, 3, 4, 5, 7, 8, 10, 13, 14, 18, 19, 23, 25, 26), 'cv_scores': array([0.15699334, 0.00116959, 0.16045845, 0.        , 0.16776007,
       0.00118203, 0.16088889, 0.        , 0.07831325, 0.07555178]), 'avg_score': 0.08023174156987828, 'feature_names': ('IP_ES_25_N1', 'Z3_IB_4_N1', 'Z1_IB_10_N1', 'Z1_IB_5_N1', 'Z3_IB_8_N1', 'ECI_IB_4_N1', 'Gs(U)_IB_12_N1', 'Gs(U)_IB_68_N1', 'Gs(U)_IB_60_N1', 'Z1_NO_sideR35_CV', 'Pb_NO_sideR35_S', 'Gs(U)_NO_ALR_SI71', 'Z3_NO_UCR_S', 'ISA_NO_NPR_S', 'IP_NO_PLR_S', 'Pb_NO_PCR_V')}, 15: {'feature_idx': (0, 1, 2, 3, 4, 5, 7, 8, 10, 13, 18, 19, 23, 25, 26), 'cv_scores': array([0.16059113, 0.00118765, 0.15829384, 0.        , 0.16604824,
       0.00114155, 0.16727273, 0.        , 0.08098592, 0.07558645]), 'avg_score': 0.08111074996456533, 'feature_names': ('IP_ES_25_N1', 'Z3_IB_4_N1', 'Z1_IB_10_N1', 'Z1_IB_5_N1', 'Z3_IB_8_N1', 'ECI_IB_4_N1', 'Gs(U)_IB_12_N1', 'Gs(U)_IB_68_N1', 'Gs(U)_IB_60_N1', 'Z1_NO_sideR35_CV', 'Gs(U)_NO_ALR_SI71', 'Z3_NO_UCR_S', 'ISA_NO_NPR_S', 'IP_NO_PLR_S', 'Pb_NO_PCR_V')}, 14: {'feature_idx': (0, 1, 2, 3, 4, 5, 7, 8, 10, 13, 19, 23, 25, 26), 'cv_scores': array([0.16018957, 0.00115075, 0.1615087 , 0.        , 0.16361974,
       0.00114155, 0.16091954, 0.        , 0.07888041, 0.07604895]), 'avg_score': 0.08034592204812366, 'feature_names': ('IP_ES_25_N1', 'Z3_IB_4_N1', 'Z1_IB_10_N1', 'Z1_IB_5_N1', 'Z3_IB_8_N1', 'ECI_IB_4_N1', 'Gs(U)_IB_12_N1', 'Gs(U)_IB_68_N1', 'Gs(U)_IB_60_N1', 'Z1_NO_sideR35_CV', 'Z3_NO_UCR_S', 'ISA_NO_NPR_S', 'IP_NO_PLR_S', 'Pb_NO_PCR_V')}, 13: {'feature_idx': (1, 2, 3, 4, 5, 7, 8, 10, 13, 19, 23, 25, 26), 'cv_scores': array([0.15469613, 0.0011655 , 0.1610338 , 0.        , 0.16213768,
       0.00113122, 0.15957447, 0.        , 0.07863248, 0.07555178]), 'avg_score': 0.07939230632578612, 'feature_names': ('Z3_IB_4_N1', 'Z1_IB_10_N1', 'Z1_IB_5_N1', 'Z3_IB_8_N1', 'ECI_IB_4_N1', 'Gs(U)_IB_12_N1', 'Gs(U)_IB_68_N1', 'Gs(U)_IB_60_N1', 'Z1_NO_sideR35_CV', 'Z3_NO_UCR_S', 'ISA_NO_NPR_S', 'IP_NO_PLR_S', 'Pb_NO_PCR_V')}, 12: {'feature_idx': (1, 2, 3, 4, 5, 7, 8, 10, 19, 23, 25, 26), 'cv_scores': array([0.15251142, 0.00113895, 0.15678776, 0.        , 0.16245487,
       0.00110988, 0.16143106, 0.        , 0.07969152, 0.07632264]), 'avg_score': 0.07914481000371829, 'feature_names': ('Z3_IB_4_N1', 'Z1_IB_10_N1', 'Z1_IB_5_N1', 'Z3_IB_8_N1', 'ECI_IB_4_N1', 'Gs(U)_IB_12_N1', 'Gs(U)_IB_68_N1', 'Gs(U)_IB_60_N1', 'Z3_NO_UCR_S', 'ISA_NO_NPR_S', 'IP_NO_PLR_S', 'Pb_NO_PCR_V')}, 11: {'feature_idx': (1, 2, 3, 4, 5, 7, 8, 10, 23, 25, 26), 'cv_scores': array([0.15237226, 0.00116822, 0.15604801, 0.        , 0.16126126,
       0.00113122, 0.16368515, 0.        , 0.07874682, 0.07639485]), 'avg_score': 0.07908078113947452, 'feature_names': ('Z3_IB_4_N1', 'Z1_IB_10_N1', 'Z1_IB_5_N1', 'Z3_IB_8_N1', 'ECI_IB_4_N1', 'Gs(U)_IB_12_N1', 'Gs(U)_IB_68_N1', 'Gs(U)_IB_60_N1', 'ISA_NO_NPR_S', 'IP_NO_PLR_S', 'Pb_NO_PCR_V')}, 10: {'feature_idx': (1, 2, 4, 5, 7, 8, 10, 23, 25, 26), 'cv_scores': array([0.15433213, 0.00112613, 0.15483871, 0.        , 0.16470588,
       0.00110742, 0.16358839, 0.        , 0.07717842, 0.07478992]), 'avg_score': 0.07916669975366768, 'feature_names': ('Z3_IB_4_N1', 'Z1_IB_10_N1', 'Z3_IB_8_N1', 'ECI_IB_4_N1', 'Gs(U)_IB_12_N1', 'Gs(U)_IB_68_N1', 'Gs(U)_IB_60_N1', 'ISA_NO_NPR_S', 'IP_NO_PLR_S', 'Pb_NO_PCR_V')}, 9: {'feature_idx': (1, 2, 4, 5, 7, 8, 10, 23, 25), 'cv_scores': array([0.15205725, 0.00112613, 0.15610652, 0.        , 0.16021127,
       0.00110742, 0.16049383, 0.        , 0.07597027, 0.07627119]), 'avg_score': 0.07833438643704513, 'feature_names': ('Z3_IB_4_N1', 'Z1_IB_10_N1', 'Z3_IB_8_N1', 'ECI_IB_4_N1', 'Gs(U)_IB_12_N1', 'Gs(U)_IB_68_N1', 'Gs(U)_IB_60_N1', 'ISA_NO_NPR_S', 'IP_NO_PLR_S')}, 8: {'feature_idx': (1, 4, 5, 7, 8, 10, 23, 25), 'cv_scores': array([0.1497373 , 0.00110497, 0.15523466, 0.        , 0.16143498,
       0.00107991, 0.16134599, 0.        , 0.07781457, 0.0762987 ]), 'avg_score': 0.07840510823337679, 'feature_names': ('Z3_IB_4_N1', 'Z3_IB_8_N1', 'ECI_IB_4_N1', 'Gs(U)_IB_12_N1', 'Gs(U)_IB_68_N1', 'Gs(U)_IB_60_N1', 'ISA_NO_NPR_S', 'IP_NO_PLR_S')}, 7: {'feature_idx': (1, 4, 7, 8, 10, 23, 25), 'cv_scores': array([0.15173026, 0.00111982, 0.1569873 , 0.        , 0.15748031,
       0.00110011, 0.16071429, 0.        , 0.07513989, 0.07380373]), 'avg_score': 0.07780757034882406, 'feature_names': ('Z3_IB_4_N1', 'Z3_IB_8_N1', 'Gs(U)_IB_12_N1', 'Gs(U)_IB_68_N1', 'Gs(U)_IB_60_N1', 'ISA_NO_NPR_S', 'IP_NO_PLR_S')}, 6: {'feature_idx': (1, 4, 7, 8, 23, 25), 'cv_scores': array([0.14762742, 0.00111483, 0.15314494, 0.        , 0.16010499,
       0.00110011, 0.15856777, 0.        , 0.07797428, 0.07481899]), 'avg_score': 0.0774453319143933, 'feature_names': ('Z3_IB_4_N1', 'Z3_IB_8_N1', 'Gs(U)_IB_12_N1', 'Gs(U)_IB_68_N1', 'ISA_NO_NPR_S', 'IP_NO_PLR_S')}, 5: {'feature_idx': (1, 4, 8, 23, 25), 'cv_scores': array([0.15057573, 0.00111235, 0.15138023, 0.        , 0.15443038,
       0.00106496, 0.15609349, 0.        , 0.07717042, 0.0752    ]), 'avg_score': 0.07670275589383031, 'feature_names': ('Z3_IB_4_N1', 'Z3_IB_8_N1', 'Gs(U)_IB_68_N1', 'ISA_NO_NPR_S', 'IP_NO_PLR_S')}, 4: {'feature_idx': (1, 8, 23, 25), 'cv_scores': array([0.15081405, 0.00107066, 0.14964789, 0.        , 0.15521628,
       0.00105485, 0.1560166 , 0.        , 0.07847708, 0.07517619]), 'avg_score': 0.07674736117637573, 'feature_names': ('Z3_IB_4_N1', 'Gs(U)_IB_68_N1', 'ISA_NO_NPR_S', 'IP_NO_PLR_S')}, 3: {'feature_idx': (8, 23, 25), 'cv_scores': array([0.15141431, 0.00104932, 0.14995641, 0.00102145, 0.15124698,
       0.00100503, 0.15453074, 0.        , 0.07354056, 0.07214121]), 'avg_score': 0.0755906012093118, 'feature_names': ('Gs(U)_IB_68_N1', 'ISA_NO_NPR_S', 'IP_NO_PLR_S')}, 2: {'feature_idx': (23, 25), 'cv_scores': array([0.14576547, 0.0010352 , 0.14811715, 0.00102354, 0.14768264,
       0.00100402, 0.14912281, 0.        , 0.0741018 , 0.07148289]), 'avg_score': 0.07393355139224522, 'feature_names': ('ISA_NO_NPR_S', 'IP_NO_PLR_S')}, 1: {'feature_idx': (23,), 'cv_scores': array([0.13187773, 0.        , 0.14174757, 0.00179856, 0.15230635,
       0.00090171, 0.13548951, 0.        , 0.08265213, 0.07889734]), 'avg_score': 0.07256709131459835, 'feature_names': ('ISA_NO_NPR_S',)}}

#--------------------------------------------------------------------------------
class BayesianPNN(algorithms.PNN):
    # ['N', 'P']
    priors = np.array([1, 1])
    def __init__(self, class_priors=None, *pargs, **kwargs):
        kwargs['batch_size'] = None
        super(BayesianPNN, self).__init__(*pargs, **kwargs)
        if class_priors:
            BayesianPNN.priors = np.array(class_priors)
            self.priors = BayesianPNN.priors

    def predict_raw(self, X):
        raw_output = super(BayesianPNN, self).predict_raw(X)
        self.classes_ = self.classes

        # Bayesian Approach to prevent over prediction of minority class.
        return (raw_output.T * self.priors).T

#--------------------------------------------------------------------------------
def import_csv(file_path):
    return pd.read_csv(file_path)

#--------------------------------------------------------------------------------
def concatenate_frames(frames):
    return pd.concat(frames)

#--------------------------------------------------------------------------------
def _create_model_instance(model):
    normalizer = Normalizer()
    
    return Pipeline([('standardizer', StandardScaler()),
                     ('undersample', RandomUnderSampler(sampling_strategy=RATIO)),
#                      ('smote', SMOTE()),
                     ('normalizer', normalizer),
                     ('pnn', model)])

#--------------------------------------------------------------------------------
def create_model_instance(class_priors=None, std=10):
    # Creates an instance of the model with the given standard deviation and
    # weighs the score of each class with the given priors. The model performs
    # SMOTE before fitting training data.
    if class_priors == None:
        class_priors = [1, 1]

    model = BayesianPNN(class_priors, std=std)

    return _create_model_instance(model)

#--------------------------------------------------------------------------------
def create_ensemble_model_instance(class_priors=None, std=10):
    # Creates an instance of the model with the given standard deviation and
    # weighs the score of each class with the given priors. The model performs
    # SMOTE before fitting training data.
    if class_priors == None:
        class_priors = [1, 1]

    model = BayesianPNN(class_priors, std=std)
    model = BaggingClassifier(base_estimator=model, n_estimators=10, n_jobs=-1)

    return _create_model_instance(model)

#--------------------------------------------------------------------------------
def remove_outliers_iqr(df):
    # Remove all outliers outside the 95% confidence interval.
    x = data.drop('class', 1)

    Q1 = x.quantile(0.25)
    Q3 = x.quantile(0.75)
    IQR = Q3 - Q1

    return df[~((x < (Q1 - 1.5 * IQR)) |
                (x > (Q3 + 1.5 * IQR))).any(axis=1)]

#--------------------------------------------------------------------------------
def remove_outliers_normal(df):
    # Remove all outliers outside the 95% confidence interval.
    x = data.drop('class', 1)
    
    return df[(np.abs(stats.zscore(x)) < 3).all(axis=1)]

#--------------------------------------------------------------------------------
def backward_elimination(model, X, y):
    # Return the optimal set of features to use for classification. Uses 10 fold
    # cross-validation to evaluate the performance of each feature set. Selects
    # feature set that achieves maximal precision.
    sbs = Sfs(model,
              k_features='parsimonious',
              forward=False,
              floating=False,
              cv=10,
              scoring=make_scorer(precision_score, greater_is_better=True, 
                                  needs_proba=False, pos_label='P'),
              n_jobs=-1,
              verbose=2)
    sbs.fit(X, y)
    print(80*'-')
    print(sbs.subsets_)
    print(80*'-')
    return sbs.k_feature_names_

#--------------------------------------------------------------------------------
def feature_extraction(X, y):
    # Return the optimal set of features to use for classification. Uses 10 fold
    # cross-validation to evaluate the performance of each feature set. Selects
    # feature set that achieves maximal precision.
    lda = PrincipalComponentAnalysis(n_discriminants=20)
    y = y.replace('N', 0).replace('P', 1)
    lda.fit(np.array(X), y)
    X_lda = lda.transform(X)
    return X_lda

#--------------------------------------------------------------------------------
def evaluate_model(model, X, y, cv=10, std=10, final=''):
    # Returns the precision at 50% recall.
    y_pred_prob = cross_val_predict(model, X, y, cv=cv, method='predict_proba')

    # Plot the Precision-Recall curve.
    pr, re, thresholds = precision_recall_curve(y, y_pred_prob[:, 1],
                                                pos_label=POSITIVE_LABEL)

    indices_above_50_re = np.where(re >= 0.5)

    pr_beyond_50 = pr[indices_above_50_re]
    thr_beyond_50 = thresholds[indices_above_50_re]
    re_beyond_50 = re[indices_above_50_re]
    
    max_pr = pr_beyond_50[np.argmax(pr_beyond_50)]
    threshold_at_max_pr = thr_beyond_50[np.argmax(pr_beyond_50)]
    re_at_max_pr = re_beyond_50[np.argmax(pr_beyond_50)]

    # Get the predicted classes of the model.
    classes = ['N', 'P']
    y_pred = np.array(['P' if y >= threshold_at_max_pr else 'N'
                       for y in y_pred_prob[:, 1]])

    # Get the confusion matrix: tn, fp, tp, fp = conf_matrix.ravel().
    conf_matrix = confusion_matrix(y, y_pred, labels=classes)
    tn, fp, fn, tp = conf_matrix.ravel()

    # Calculate precision's standard deviation.
    pr_std = get_precision_std(tp, tp + fp)
    
    print('-' * 30 + 'std_param = {}'.format(std) + 30 * '-')

    print('TP {}, FP {}'.format(tp, fp))
    print('FN {}, TN {}'.format(fn, tn))

    print('Threshold: {}'.format(threshold_at_max_pr))
    print('Recall: {}'.format(re_at_max_pr))
    print('Precision: {}'.format(max_pr))
    print('Standard deviation: {}'.format(pr_std))

    # Compute accuracy (for meta-learning).
    acc_score = accuracy_score(y, y_pred)
    print("Accuracy : {}".format(acc_score))

    plt.figure()
    plt.step(re, pr, color='b', alpha=0.8, where='post')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve for {} Features with {} std'
              .format(X.shape[1], std))
    plt.savefig(os.path.join(BASE_DIR, 'Pr_Re_{}_features_{}_std_{}.jpg'.format(
        X.shape[1], std, final)))

    print('-' * 80)
    return max_pr, pr_std

#--------------------------------------------------------------------------------
def get_precision_std(tp, n):
    p = float(tp) / n
    variance_of_sum = p * (1 - p) / n
    std = variance_of_sum ** 0.5
    return std

#--------------------------------------------------------------------------------
def test_run(y_pred_prob, y_actual):
    # Plot the Precision-Recall curve.
    pr, re, thresholds = precision_recall_curve(y_actual, y_pred_prob[:, 1],
                                                pos_label=POSITIVE_LABEL)

    indices_above_50_re = np.where(re >= 0.5)

    pr_beyond_50 = pr[indices_above_50_re]
    thr_beyond_50 = thresholds[indices_above_50_re]
    re_beyond_50 = re[indices_above_50_re]
    
    max_pr = pr_beyond_50[np.argmax(pr_beyond_50)]
    threshold_at_max_pr = thr_beyond_50[np.argmax(pr_beyond_50)]
    re_at_max_pr = re_beyond_50[np.argmax(pr_beyond_50)]
    
    print('Precision: {}'.format(max_pr))

#--------------------------------------------------------------------------------
if __name__ == "__main__":
    csv_path_training = os.path.join(BASE_DIR, CSV_FILE_TRAINING)
    csv_path_calibration = os.path.join(BASE_DIR, CSV_FILE_CALIBRATION)
    csv_path_blind = os.path.join(BASE_DIR, CSV_FILE_BLIND)
    
    # All of our data collected.
    data = concatenate_frames([import_csv(csv_path_training),
                               import_csv(csv_path_calibration)])
    
    data_blind = import_csv(csv_path_blind)
    X_blind = data_blind
#     y_blind = data_blind['class']
    
    # Drop id column from data.
    data = data.drop(['id'], 1)

    # Segregate the negative and positive populations.
    negative_data = data.loc[data['class'] == 'N']
    positive_data = data.loc[data['class'] == 'P']

    # Calculate priors.
    negative_prior = len(negative_data.index) / (1. * len(data.index))
    positive_prior = len(positive_data.index) / (1. * len(data.index))
    priors = [1, 1]
    
    # Separate features from their corresponding class.
    X = data.drop(['class'], 1)
    y = data['class']

    selected_features = list(ALL_SUBSETS_47[14]['feature_names'])
    print('Features selected: {}'.format(selected_features))

    X_new = X[selected_features]
    X_blind = X_blind[selected_features]

    # Final model.
    MAX_PARAM = 7.6
    model = create_model_instance(priors, MAX_PARAM)
    model.fit(X_new, y)
    y_labels = model.predict_proba(X_blind)
    y_proba_p = np.array(y_labels[:,1]).T
    OUTPUT_FILE = 'Blind_Test_scores.txt'
    np.savetxt(OUTPUT_FILE, y_proba_p, delimiter=',')
    print(y_proba_p)
    
    data_blind.to_csv()
#     test_run(y_labels, y_blind)
#     np.savetxt("final_scores.csv", y_proba_p, delimiter=",")


# In[ ]:




