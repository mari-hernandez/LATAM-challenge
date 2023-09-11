import os
import pickle
import pandas as pd

from typing import Tuple, Union, List
import numpy as np
from datetime import datetime
import pandas as pd
from sklearn.utils import shuffle
import warnings

warnings.filterwarnings('ignore')

def get_min_diff(data):
    fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
    fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
    min_diff = ((fecha_o - fecha_i).total_seconds())/60
    return min_diff

class DelayModel:

    def __init__(self):
        """
        Initialize the DelayModel with a pre-trained XGBoost model.
        """
        self._model = self.load_model()
        self.top_10_features = self._model.get_booster().feature_names

        
    def load_model(self):
        """
        Load the model from the pickle file.
        """
        try:
            model_path = os.path.join('challenge', 'xgboost_model.pkl')
            return pickle.load(open(model_path, 'rb'))
        except FileNotFoundError:
            raise FileNotFoundError("The XGBoost model file was not found.")
        except Exception as e:
            raise Exception(f"Error loading the XGBoost model: {str(e)}")

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """ 
        # Check if required columns are present
        required_columns = ['OPERA', 'MES', 'TIPOVUELO']
        if target_column:
            required_columns += ['Fecha-O', 'Fecha-I']

        if not set(required_columns).issubset(data.columns):
            raise ValueError(f"Missing required columns: {', '.join(set(required_columns) - set(data.columns))}")

        if target_column:  # Training mode
            data['min_diff'] = (pd.to_datetime(data['Fecha-O']) - pd.to_datetime(data['Fecha-I'])).dt.total_seconds() / 60
            threshold_in_minutes = 15
            data[target_column] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)
            data = shuffle(data[['OPERA', 'MES', 'TIPOVUELO', target_column]])

        features = pd.concat([
                pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
                pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'), 
                pd.get_dummies(data['MES'], prefix = 'MES')], 
                axis = 1
            )
        
        # Select top 10 features
        features = features[self.top_10_features]

        if target_column:
            target = data[[target_column]]
            return features, target
        else:
            return features

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        self._model.fit(features, target)
        self._model.get_booster().feature_names

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        predictions = self._model.predict(features)
        return predictions.tolist()
