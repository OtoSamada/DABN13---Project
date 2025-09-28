# Data imports
import pandas as pd
import numpy as np
import math

# Visualization imports
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
plt.rcParams['figure.figsize'] = [8, 4]


# ML Imports
from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
import missingno as msno
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import shap

from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix
import pickle

import warnings
warnings.filterwarnings("ignore")