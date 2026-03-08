from src.helpers.initiate_stage import initiate_file
from src.storage.minio_client import MinIOClient
from src.helpers.exception_handling import MyException
from src.helpers.load_save import log_dataset_profile, load_data

from dotenv import load_dotenv
load_dotenv()
import os
import sys
import pandas as pd 

class DataIngestion:
    def __init__(self, params, logger) -> None:
        self.params = params
        self.logger = logger

        self.object_key = self.params['object_key']
        self.local_path = self.params['local_path']

        try : 
            self.client = MinIOClient(endpoint_url=os.getenv("END_POINT_URL", ''), 
                                    access_key=os.getenv('ACCESS_KEY', ''), 
                                    secret_key=os.getenv('SECRET_KEY', ''), 
                                    bucket_name=self.params['bucket_name'])
            self.logger.debug('CLIENT CONNECTION BUILT SUCCESSFULLY')
        except Exception as e : 
            raise MyException(e, sys, self.logger)
        
    def download_file(self): 
        try : 
            if self.client.object_exists(object_key=self.object_key):
                self.client.download_file(self.object_key, 
                                        local_path=self.local_path)
                self.logger.info('File Downloaded Successfully at %s', self.local_path)
                return self.local_path
            else : 
                self.logger.warning("%s object doesn't exists", self.object_key)
                raise FileNotFoundError(f"{self.object_key} not found in bucket")
        except Exception as e : 
            raise MyException(e, sys, self.logger)
    
    def run(self): 
        try:
            os.makedirs(os.path.dirname(self.local_path), exist_ok=True)
            dataset_path = self.download_file()
            self.logger.info("Data Ingestion Completed")

            df = load_data(file_path=dataset_path, logger=self.logger)
            log_dataset_profile(df=df, 
                                dataset_name=dataset_path, 
                                output_path=self.params['output_profile_path'], 
                                bucket_name=self.params['bucket_name'], 
                                object_key=self.object_key)
            self.logger.info('DATA INGESTION PROFILE LOG IS SAVED AT %s', self.params['output_profile_path'])
        except Exception as e: 
            raise MyException(e, sys, self.logger)

if __name__ == "__main__": 
    params, logger = initiate_file(params_file_path="config/training.yaml", 
                                   component_name='DataIngestion')
    
    data_ingestion = DataIngestion(params=params, logger=logger)
    data_ingestion.run()