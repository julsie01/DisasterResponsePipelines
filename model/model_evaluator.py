import pandas as pd
from pprint import pprint
from time import time
import logging
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


class ModelEvaluation:

    def display_results(self, test, preds, labels):
        """
        Input: test: test dataset, preds: model predictions, labels: set of labels for the predictions
        Output: print out of the classification report
        """
        item_count = len(labels)
        df_report = pd.DataFrame()
        # output_dict=True
        #if using scikit learn latest version its possible to get a dictionary as output,
        #however removing as the terminal is using an older version. 

        for i in range(1,item_count):
            
            classification_rpt = classification_report(np.asarray(preds[:i]), np.asarray(test[:i]))
            
            
        print(classification_rpt)   
            #classification_report_df = pd.DataFrame.from_dict(classification_rpt)
            #classification_report_df['label'] = labels[i]
            #df_report = df_report.append(classification_report_df)
            
        
        
        #return df_report