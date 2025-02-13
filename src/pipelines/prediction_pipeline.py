import sys,os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd


class PredictionPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')
            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)
            data_scaled=preprocessor.transform(features)
            pred=model.predict(data_scaled)
            return pred
        except Exception as e:
            raise CustomException(e,sys)         
                  
class CustomData:
    def __init__(self,
                 age=float,
                 sex=str,
                 on_thyroxine=str,
                 sick=str,
                 pregnant=str,
                 I131_treatment=str,
                 query_hypothyroid=float,
                 goitre=str,
                 psych=str,
                 TSH=float,
                 T3=float,
                 TT4=float,
                 T4U=float,
                 FTI=float,
                 referral_source_SVHC=str,
                 referral_source_other=str
                 
                 ):
        
       self.age=age
       self.sex=sex
       self.on_thyroxine=on_thyroxine
       self.sick=sick
       self.pregnant=pregnant
       self.I131_treatment=I131_treatment
       self.query_hypothyroid=query_hypothyroid
       self.goitre=goitre
       self.psych=psych
       self.TSH=TSH
       self.T3=T3
       self.TT4=TT4
       self.T4U=T4U
       self.FTI=FTI
       self.referral_source_SVHC=referral_source_SVHC
       self.referral_source_other=referral_source_other
    def get_data_as_dataframe(self):
        try:
               CustomData={
                   'age':[self.age],
                   'sex':[self.sex],
                   'on_thyroxine':[self.on_thyroxine],
                   'sick':[self.sick],
                   'pregnant':[self.pregnant],
                   'I131_treatment':[self.I131_treatment],
                   'query_hypothyroid':[self.query_hypothyroid],
                   'goitre':[self.goitre],
                   'psych':[self.psych],
                   'TSH':[self.TSH],
                   'T3':[self.T3],
                   'TT4':[self.TT4],
                   'T4U':[self.T4U],
                   'FTI':[self.FTI],
                   'referral_source_SVHC':[self.referral_source_SVHC],
                   'referral_source_other':[self.referral_source_other]
                   
                   }
               df=pd.DataFrame(CustomData)
               logging.info('DataFrame Gathered')
               return df
        except Exception as e:
            logging.info('dataframe not gathered due to error in prediction pipeline')
            raise CustomException(e ,sys)
        

        
