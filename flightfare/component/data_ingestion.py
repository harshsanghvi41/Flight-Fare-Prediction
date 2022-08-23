from flightfare.exception import FlightFareException
from flightfare.logger import logging
from flightfare.entity.config_entity import DataIngestionConfig
from flightfare.entity.artifact_entity import DataIngestionArtifact
import os, sys
import pymongo
import pandas as pd
from flightfare.constant import *

class DataIngestion:
    def __init__(self, data_ingestion_config : DataIngestionConfig):
        try:
            logging.info(f"{'='*20} DataIngestion log started. {'='*20}")
            self.data_ingestion_config = data_ingestion_config

        except Exception as e:
            raise FlightFareException(e,sys) from e
    

    def get_train_test_data_from_mongodb(self):
        try:
            self.client = pymongo.MongoClient("mongodb+srv://mongodb:mongodb@cluster0.akfky.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
            db = self.client['harsh']
            logging.info("Database Authenticated!")

            col_train = db['FlightFareTrain']
            col_test = db['FlightFareTest']
            logging.info("Train Collection Created/Found!")
            logging.info("Test Collection Created/Found!")

            cursor_train = col_train.find()
            cursor_test = col_test.find()

            mongo_docs_train = list(cursor_train)
            mongo_docs_test = list(cursor_test)

            df_train = pd.DataFrame(mongo_docs_train, columns=["Airline", "Date_of_Journey", "Source", "Destination", "Route", "Dep_Time", "Arrival_Time", "Duration", "Total_Stops", "Additional_Info", "Price"])
            logging.info("Storing data into dataframe : df_train")

            df_test = pd.DataFrame(mongo_docs_test, columns=["Airline", "Date_of_Journey", "Source", "Destination", "Route", "Dep_Time", "Arrival_Time", "Duration", "Total_Stops", "Additional_Info"])
            logging.info("Storing data into dataframe : df_test")


            file_name = 'Flight_Fare.csv'
            logging.info(f"File Name : {file_name}")

            train_file_path = os.path.join(self.data_ingestion_config.ingested_train_dir, file_name)
            logging.info(f"Train File Path : {train_file_path}")
            test_file_path = os.path.join(self.data_ingestion_config.ingested_test_dir, file_name)
            logging.info(f"Test File Path : {test_file_path}")
            
            
            os.makedirs(self.data_ingestion_config.ingested_train_dir, exist_ok=True)
            logging.info(f"Exporting training datset to file: [{train_file_path}]")
            df_train.to_csv(train_file_path,index=False)

            os.makedirs(self.data_ingestion_config.ingested_test_dir, exist_ok=True)
            logging.info(f"Exporting testing datset to file: [{test_file_path}]")
            df_test.to_csv(test_file_path,index=False)

            
            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path = train_file_path,
                test_file_path = test_file_path,
                is_ingested = True,
                message = f"Data ingestion completed successfully.")


            logging.info(f"Data Ingestion artifact:[{data_ingestion_artifact}]")
            return df_train, df_test, data_ingestion_artifact

        except Exception as e:
            raise FlightFareException(e)



    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            self.get_train_test_data_from_mongodb()
            return self.get_train_test_data_from_mongodb()

        except Exception as e:
            raise FlightFareException(e,sys) from e

    def __del__(self):
        logging.info(f"{'='*20}Data Ingestion log completed.{'='*20} \n")
