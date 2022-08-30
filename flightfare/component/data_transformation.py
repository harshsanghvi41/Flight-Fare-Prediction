from flightfare.exception import FlightFareException
from flightfare.logger import logging
import os,sys
from flightfare.entity.config_entity import DataTransformationConfig 
from flightfare.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact
from sklearn import preprocessing
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import pandas as pd
from flightfare.constant import *
from flightfare.util.util import *


class FeatureGenerator(BaseEstimator, TransformerMixin):

    def __init__(self, Airline_ix = 0,
                 Date_of_Journey_ix = 1,
                 Source_ix = 2,
                 Destination_ix = 3,
                 Route_ix = 4,
                 Dep_Time_ix = 5,
                 Arrival_Time_ix = 6,
                 Duration_ix = 7, columns = None):
       
        try:
            self.columns = columns
            if self.columns is not None:
                Airline_ix = self.columns.index(COLUMN_AIRLINE)
                Date_of_Journey_ix = self.columns.index(COLUMN_DATE_OF_JOURNEY)
                Dep_Time_ix = self.columns.index(COLUMN_DEP_TIME)
                Source_ix = self.columns.index(COLUMN_SOURCE)
                Destination_ix = self.columns.index(COLUMN_DESTINATION)
                Route_ix = self.columns.index(COLUMN_ROUTE)
                Duration_ix = self.columns.index(COLUMN_DURATION)
                Arrival_Time_ix = self.columns.index(COLUMN_ARRIVAL_TIME)

            """self.Journey_day = Journey_day
            self.Journey_month = Journey_month
            self.Arrival_hour = Arrival_hour
            self.Arrival_min = Arrival_min
            self.Dep_hour = Dep_hour
            self.Dep_min = Dep_min"""

            self.Airline_ix = Airline_ix
            self.Date_of_Journey_ix = Date_of_Journey_ix
            self.Dep_Time_ix = Dep_Time_ix
            self.Source_ix = Source_ix
            self.Destination_ix = Destination_ix
            self.Route_ix = Route_ix
            self.Duration_ix = Duration_ix
            self.Arrival_Time_ix = Arrival_Time_ix

        except Exception as e:
            raise FlightFareException(e, sys) from e


    def fit(self, X, y=None):
        return self


    def transform(self, X, y=None):
        try:
            Journey_day = X[:, pd.to_datetime(self.Date_of_Journey_ix, format="%d/%m/%Y").day] 
            Journey_month = X[:, pd.to_datetime(self.Date_of_Journey_ix, format = "%d/%m/%Y").month]
            logging.info("Journey_day and month")

            Dep_hour = X[:, pd.to_datetime(self.Dep_Time_ix).hour]
            Dep_min =  X[:, pd.to_datetime(self.Dep_Time_ix).minute]
            logging.info("Dep_hour and min")

            Arrival_hour = X[:, pd.to_datetime(self.Arrival_Time_ix).hour]
            Arrival_min =  X[:, pd.to_datetime(self.Arrival_Time_ix).minute]                 
            logging.info("Arrival_hour and min")


            duration = list(self.Duration_ix)
            logging.info("duration")

            for i in range(len(duration)):
                if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
                    logging.info("enter into for i in range(len(duration))")
                    if "h" in duration[i]:
                        duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
                    else:
                        duration[i] = "0h " + duration[i]           # Adds 0 hour


            Duration_hours = []
            Duration_mins = []
            for i in range(len(duration)):
                Duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
                logging.info("Duration_hours")
                Duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1])) 
                logging.info("Duration_mins")

            generated_feature = np.c_[X, Journey_day, Journey_month, Arrival_hour, Arrival_min, Dep_hour, Dep_min, Duration_hours, Duration_mins]
            logging.info("generated_feature")

            return generated_feature

        except Exception as e:
            raise FlightFareException(e, sys) from e


class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact):
                    
        try:
            logging.info(f"{'=' * 30}Data Transformation log started.{'=' * 30} ")
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.data_tranformation_config = data_transformation_config

        except Exception as e:
                raise FlightFareException(e,sys) from e

        
    def get_data_transformer_object(self) -> ColumnTransformer:
        try:
            schema_file_path = self.data_validation_artifact.schema_file_path

            dataset_schema = read_yaml_file(file_path = schema_file_path)

            #numerical_columns = dataset_schema[NUMERICAL_COLUMN_KEY]
            categorical_columns = dataset_schema[CATEGORICAL_COLUMN_KEY]
            label_encoder_cat_columns = dataset_schema[ONE_CATEGORICAL_COLUMN_KEY]

            #num_pipeline = Pipeline(steps=[])
            
            cat_pipeline = Pipeline(steps=[
                ('feature_generator', FeatureGenerator(
                    Journey_day = self.data_tranformation_config.Journey_day,
                    Journey_month = self.data_tranformation_config.Journey_month,
                    Arrival_hour = self.data_tranformation_config.Arrival_hour,
                    Arrival_min = self.data_tranformation_config.Arrival_min,
                    Dep_hour = self.data_tranformation_config.Dep_hour,
                    Dep_min = self.data_tranformation_config.Dep_min,
                    Duration_hours = self.data_tranformation_config.Duration_hours,
                    Duration_mins = self.data_tranformation_config.Duration_mins,
                    columns = categorical_columns
                )),
                ('one_hot_encoder', OneHotEncoder())
            ]
            )

            label_encoder_cat_pipeleine = Pipeline(steps=[
                ('label_encoder', LabelEncoder())
            ])

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {label_encoder_cat_columns}")

            preprocessing = ColumnTransformer([
                #('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns),
                ('label_encoder_cat_pipeleine', label_encoder_cat_pipeleine, label_encoder_cat_columns)
            ])
            return preprocessing

        except Exception as e:
                raise FlightFareException(e,sys) from e


    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info(f"Obtaining preprocessing object.")
            preprocessing_obj = self.get_data_transformer_object()

            logging.info(f"Obtaining training and test file path.")
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            schema_file_path = self.data_validation_artifact.schema_file_path

            logging.info(f"Loading training and test data as pandas dataframe.")
            train_df = load_data(file_path = train_file_path, schema_file_path = schema_file_path)
            test_df = load_data(file_path = test_file_path, schema_file_path = schema_file_path)

            schema = read_yaml_file(file_path = schema_file_path)

            target_column_name = schema[TARGET_COLUMN_KEY]

            logging.info(f"Splitting input and target feature from training and testing dataframe.")
            input_feature_train_df = train_df.drop(columns = [target_column_name, "Airline", "Source", "Destination", "Date_of_Journey", "Dep_Time", "Arrival_Time", "Duration", "Route", "Additional_Info"], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns = [target_column_name, "Airline", "Source", "Destination", "Date_of_Journey", "Dep_Time", "Arrival_Time", "Duration", "Route", "Additional_Info"], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]


            transformed_train_dir = self.data_tranformation_config.transformed_train_dir
            transformed_test_dir = self.data_tranformation_config.transformed_test_dir


            train_file_name = os.path.basename(train_file_path).replace(".csv",".npz")
            test_file_name = os.path.basename(test_file_path).replace(".csv",".npz")


            transformed_train_file_path = os.path.join(transformed_train_dir, train_file_name)
            transformed_test_file_path = os.path.join(transformed_test_dir, test_file_name)


            logging.info(f"Saving transformed training and testing array.")
            save_numpy_array_data(file_path = transformed_train_file_path, array = train_arr)
            save_numpy_array_data(file_path = transformed_test_file_path, array = test_arr)


            preprocessing_obj_file_path = self.data_tranformation_config.preprocessed_object_file_path
            

            logging.info(f"Saving preprocessing object.")
            save_object(file_path = preprocessing_obj_file_path, obj = preprocessing_obj)


            data_transformation_artifact = DataTransformationArtifact(
                    is_transformed = True,
                    message = "Data transformation successfull.",
                    transformed_train_file_path = transformed_train_file_path,
                    transformed_test_file_path = transformed_test_file_path,
                    preprocessed_object_file_path = preprocessing_obj_file_path)


            logging.info(f"Data transformationa artifact: {data_transformation_artifact}")
            return data_transformation_artifact

        except Exception as e:
            raise FlightFareException(e,sys) from e

    def __del__(self):
        logging.info(f"{'='*30}Data Transformation log completed.{'='*30} \n")
