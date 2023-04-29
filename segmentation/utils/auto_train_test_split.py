from sklearn.model_selection import train_test_split
import pandas as pd
import os
import sys

print('path',os.getcwd())


class AutoTrainTestSplit():
    def __init__(self,info_dict):
        print('inside initialize')
        if info_dict.get("train_test_file",'off') == 'on':
            print('split by file name')
            split_by_file_name_info_keys = ['dataset_path','train_file','test_file',"file_type",]
            split_by_file_name_info_dict = {key:info_dict[key] for key in split_by_file_name_info_keys}
            self.train_test_split_by_file_name(split_by_file_name_info_dict)
            
        else:
            print('split by percentage')
            split_by_percentage_info_keys = ['dataset_path','file_name',"file_type","test_split"]
            split_by_perc_info_dict = {key:info_dict[key] for key in split_by_percentage_info_keys}
            self.train_test_split_by_percentage(split_by_perc_info_dict)

    def file_to_df(self,path,file_type):
        if file_type.lower() == 'csv':
            df = pd.read_csv(path)
        elif file_type.get('file_type').lower() == 'excel':
            df = pd.read_csv(path)
        else:
            raise Exception('Only csv and excel files are supported please choose from those 2')
        return df
        
    def train_test_split_by_percentage(self,info_dict):
        file_path = os.path.join('dataset',info_dict.get('dataset_path'),info_dict.get('file_name'))
        print('file_path',file_path)
        full_path = os.path.join(os.getcwd(),file_path)
        print('full_path',full_path)
        df = self.file_to_df(full_path,info_dict.get('file_type'))
        self.train,self.test = train_test_split(df,test_size=float(info_dict.get('test_split')),random_state=42)
        
    def train_test_split_by_file_name(self,info_dict):
        
        train_file_path = os.path.join(os.getcwd(),'dataset',info_dict.get('dataset_path'),info_dict.get('train_file'))
        print(train_file_path)
        test_file_path = os.path.join(os.getcwd(),'dataset',info_dict.get('dataset_path'),info_dict.get('test_file'))
        print(test_file_path)
        
        self.train = self.file_to_df(train_file_path,info_dict.get('file_type'))
        self.test = self.file_to_df(test_file_path,info_dict.get('file_type'))
    
    
    
    def __call__(self):
        print('call')
        # one hot encoded data
        self.encoded_train = pd.get_dummies(self.train)
        self.encoded_test = pd.get_dummies(self.test)
        return self.train,self.test,self.encoded_train,self.encoded_test