
import logging 
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

logger= logging.getLogger(__name__)


def create_preprocessor()-> Pipeline :
    logger.info(' creating feature engineering pipeline')
    numerical_featurs=['step','amount','age']

    categorical_features=['gender','category','merchant','zipcodeOri','zipMerchant']

    numerical_pipeline= Pipeline(
        steps=[('scaler',StandardScaler())]
    )
    categorical_pipeline=Pipeline(steps=[('encoder',OneHotEncoder(handle_unknown='ignore'))])
    preprocessor=ColumnTransformer(
        transformers=[('num',numerical_pipeline,numerical_featurs),
                      ('cat',categorical_pipeline,categorical_features)],
                      remainder='drop'
    )
    logger.info(' features created succefully')
    return preprocessor