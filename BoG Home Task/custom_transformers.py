import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_transformer


class ColumnRenamer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X, columns=self.columns)
        return X

class MonthConverter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if 'arrival_date_month' in X.columns:
            X = X.copy()
            X["arrival_date_month"] = X["arrival_date_month"].replace({
                "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
                "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12,})
        return X

class CountryToContinent(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if 'country' in X.columns:
            X = X.copy()
            conditions = [
                X['country'].isin(['PRT', 'GBR', 'ESP', 'IRL', 'FRA', 'ROU', 'NOR', 'POL', 'DEU', 'BEL', 'CHE', 'GRC', 'ITA', 'NLD', 'DNK', 'RUS', 'SWE', 'EST', 'CZE', 'FIN', 'LUX', 'SVN', 'ALB', 'UKR', 'SMR', 'LVA', 'SRB', 'AUT', 'BLR', 'LTU', 'TUR', 'HRV', 'AND', 'GIB', 'URY', 'JEY', 'GGY', 'SVK', 'HUN', 'BIH', 'BGR', 'CIV', 'MKD', 'ISL', 'MLT', 'IMN', 'LIE', 'MNE', 'FRO']),
                X['country'].isin(['USA', 'CAN', 'MEX', 'PRI', 'JAM', 'CYM', 'ZMB', 'KNA', 'TWN', 'GLP', 'BRB', 'DMA', 'PYF', 'ASM', 'UMI', 'VGB']),
                X['country'].isin(['ARG', 'BRA', 'CHL', 'URY', 'COL', 'VEN', 'SUR', 'ECU', 'PER', 'BOL', 'PRY', 'GUY']),
                X['country'].isin(['CHN', 'IND', 'KOR', 'HKG', 'IRN', 'ARE', 'GEO', 'ARM', 'ISR', 'PHL', 'SEN', 'IDN', 'JPN', 'KWT', 'MDV', 'THA', 'MYS', 'LKA', 'SGP', 'MMR', 'UZB', 'KAZ', 'BDI', 'SAU', 'VNM', 'TJK', 'PAK', 'IRQ', 'NPL', 'BGD', 'QAT', 'JAM', 'MAC', 'TGO', 'RWA', 'KHM', 'SYR', 'JAM', 'JEY']),
                X['country'].isin(['MOZ', 'MAR', 'AGO', 'ZAF', 'EGY', 'NGA', 'KEN', 'ZWE', 'DZA', 'TUN', 'CMR', 'CIV', 'COM', 'UGA', 'GAB', 'GNB', 'MRT', 'DJI', 'STP', 'KEN', 'TZA', 'SDN', 'LBR', 'LKA']),
                X['country'].isin(['AUS', 'FJI', 'NZL', 'PNG', 'PLW', 'NCL', 'KIR', 'FSM', 'SLB', 'NRU', 'TUV', 'TON', 'WSM', 'ASA', 'ATF', 'FJI'])
            ]
            choices = [
                'Europe',
                'North America',
                'South America',
                'Asia',
                'Africa',
                'Oceania'
            ]
            X['continent'] = np.select(conditions, choices, default='Others')
        return X

class DateOfMonthTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if 'arrival_date_day_of_month' in X.columns:
            X = X.copy()
            conditions = [
                X['arrival_date_day_of_month'] < 11,
                X['arrival_date_day_of_month'] < 21
            ]
            choices = [
                'BoM', # Beginning of Month
                'MoM', # Middle of Month
            ]
            X['arrival_date_day_of_month']  = np.select(conditions, choices, default='EoM') # End of Month
        return X

class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        existing_columns = [col for col in self.columns_to_drop if col in X.columns]
        return X.drop(columns=existing_columns, axis=1)

class ADRNegativeReplacer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.median_adr = X['adr'].median() if 'adr' in X.columns else None
        return self

    def transform(self, X, y=None):
        if 'adr' in X.columns and self.median_adr is not None:
            X = X.copy()
            X['adr'] = np.where(X['adr'] < 0, self.median_adr, X['adr'])
        return X

class AdultsChildrenBabiesFilter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        if 'adults' in X.columns:
            X = X[X['adults'] > 0]
        if 'children' in X.columns:
            X = X[X['children'] < 10]
        if 'babies' in X.columns:
            X = X[X['babies'] < 10]
        return X

class ADRFilter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if 'adr' in X.columns:
            X = X[X['adr'] < 5000]
        return X

def clean_data_pipeline():
    numeric_features = ['adr', 'adults', 'children', 'babies']
    categorical_features = ['arrival_date_day_of_month', 'arrival_date_month', 'country', 'assigned_room_type', 'meal', 'market_segment', 'distribution_channel', 'continent']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        #('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = make_column_transformer(
        (numeric_transformer, numeric_features),
        (categorical_transformer, categorical_features)
    )

    pipeline_steps = [
        ('month_converter', MonthConverter()),
        ('country_to_continent', CountryToContinent()),
        ('date_of_month_transformer', DateOfMonthTransformer()),
        ('column_dropper', ColumnDropper(columns_to_drop=["reservation_status", 'reservation_status_date', 'agent', 
                                                          'company', 'arrival_date_year'])),
        ('adr_negative_replacer', ADRNegativeReplacer()),
        ('adults_children_babies_filter', AdultsChildrenBabiesFilter()),
        ('adr_filter', ADRFilter()),
        ('preprocessor', preprocessor)
    ]

    pipeline = Pipeline(steps=pipeline_steps)

    return pipeline