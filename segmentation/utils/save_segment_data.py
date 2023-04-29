import pandas as pd

class SaveSegmentData():

    def __init__(self) -> None:
        self.segment_count = 0
        self.segmented_data_dict = dict()

    def get_feature_and_category_condition(self,condition):
        split_key_value = condition.split(',')
        condition_dict = {item.split(':')[0]:item.split(':')[-1] for item in split_key_value}
        return condition_dict

    def create_and_save_segment(self,train_df,condition_dict):
        if '>' in  condition_dict.get('categories'):
            condition = int(condition_dict.get('categories').split('>')[-1])
            column = condition_dict.get('feature')
            print(__name__,condition,column)
            segmented_index = train_df[train_df[column]>condition].index
            remaining_index =train_df[train_df[column]<=condition].index
        elif '<=' in  condition_dict.get('categories'):
            condition = int(condition_dict.get('categories').split('<=')[-1])
            column = condition_dict.get('feature')
            print(__name__,condition,column)
            segmented_index = train_df[train_df[column]<=condition].index
            remaining_index = train_df[train_df[column]>condition].index
        else:
            condition = condition_dict.get('categories')
            column = condition_dict.get('feature')
            print(__name__,condition,column)
            segmented_index = train_df[train_df[column]==condition].index
            remaining_index = train_df[train_df[column]!=condition].index
        print(__name__,remaining_index)
        # self.new_train = remaining_index
        self.segmented_data_dict['segmented_index'+str(self.segment_count)] = segmented_index
        self.segmented_data_dict['remaining_index'] = remaining_index
        print(self.segmented_data_dict)
        self.segment_count += 1

    # def __call__(self):
    #     return self.segmented_data_dict
        





