import pandas as pd
import numpy as np

class AutoSegmentation:
    def __init__(self,df,target,event,no_of_category=10,no_of_bucket=10):
        self.df = df
        self.no_of_category = int(no_of_category)
        self.target = target
        if event.isdigit():
            self.event = int(event)
        else:
            self.event = event
        self.no_of_bucket = int(no_of_bucket)
        self.segmentation_df = pd.DataFrame()
        self.column_schema = ['feature','categories','event_rate','volume','event_ratio_volume']
    def object_to_string(self):
        # dtype_change
        print(type(self.df))
        object_column = self.df.select_dtypes(include='object').columns
        self.df[object_column] = self.df[object_column].astype(pd.StringDtype())
        
        

    def get_feature_to_segment_on(self):
        '''
            return a dictionary
        '''
        self.object_to_string() # will make sure all objects datatypes are converted to string
        col_dtypes = self.df.dtypes.apply(lambda x: x.name).to_dict()
        # get the columns we need to perform segmentation on
        segment_col = dict()
        for col,val in col_dtypes.items():
            if (val == 'string' and self.df[col].nunique() <= self.no_of_category) or val == 'bool':
                segment_col[col] = val
            elif 'int' in val or 'float' in val:
                segment_col[col] = val
            
        return segment_col
    

    def start_segmentation(self):
        self.segment_col = self.get_feature_to_segment_on()
        del self.segment_col[self.target] # remove target from dictionary
        # print('segment_col',self.segment_col)
        result_df = pd.DataFrame([],columns=self.column_schema)
        for col,val in self.segment_col.items():
            if val == 'string' or val == 'bool': # i.e its categorical value
                self.df[col] = self.df[col].fillna('OTH') #will handle na values in categorical features by replacing it with OTH
                # print(col,':-',val)
                event_vol_df = self.calculate_event_ratio_category(col,self.target)
            elif 'int' in val or 'float' in val:
                # print(col,":-",val)
                event_vol_df = self.calculate_event_and_volume_for_numerical(col,self.target)
            # result_df = result_df.append(event_vol_df)
            result_df = pd.concat([result_df,event_vol_df],axis=0)
            
        self.segmentation_df = result_df





    def calculate_volume_for_categorical(self,col,target):
        temp = (self.df.groupby([col],dropna=False)[target].count()/self.df[target].count()).reset_index()
        volume = dict(zip(temp[col],temp[target]))
        return volume



    def calculate_event_rate_for_categorical(self,col,target):
        temp = (self.df.groupby([col,target])[target].count()).to_frame()
        temp.columns = ['count']
        temp = temp.reset_index()
        temp = temp[temp[target]==self.event].get([col,'count'])
        temp['event_rate'] = temp['count']/temp['count'].sum()
        temp= temp.sort_values('event_rate', ascending= False)
        event_rate = dict(zip(temp[col],temp['event_rate']))
        return event_rate
    
    def calculate_event_ratio_category(self,col,target):
        output = dict()
        categorical_volume = self.calculate_volume_for_categorical(col,target)
        categorical_event_rate = self.calculate_event_rate_for_categorical(col,target)
        output[col] = {'volume':categorical_volume,'event_rate':categorical_event_rate}
        # for key in output.keys():
        list_df = []
        categories = list(set(list(output[col]['volume']) + list(output[col]['event_rate'])))
        # print(categories)
        for cat in categories:
            if cat not in output[col]['event_rate']:
                output[col]['event_rate'][cat] = 0
            get_cat_event_rate = output[col]['event_rate'].get(cat,0)
            get_cat_volume = output[col]['volume'][cat]
            event_ratio_volume = get_cat_event_rate/get_cat_volume
            list_df.append([col,cat,get_cat_event_rate,get_cat_volume,event_ratio_volume])
        
        output_df = pd.DataFrame(data=list_df,columns=self.column_schema)
        return output_df
    


    def ks(self,df=None,target=None, prob=None,no_bucket = 10):
        data = df.copy()
        data['target0'] = 1 - data[target]
        data['bucket'] = pd.qcut(data[prob], no_bucket, duplicates = 'drop')
        grouped = data.groupby('bucket', as_index = False)
        kstable = pd.DataFrame()
        kstable['min_col'] = grouped.min()[prob]
        kstable['max_col'] = grouped.max()[prob]
        kstable['events']   = grouped.sum()[target]
        kstable['nonevents'] = grouped.sum()['target0']
        kstable = kstable.sort_values(by="min_col", ascending=False).reset_index(drop = True)
        kstable['event_rate'] = (kstable.events / data[target].sum()).apply('{0:.2%}'.format)
        kstable['nonevent_rate'] = (kstable.nonevents / data['target0'].sum()).apply('{0:.2%}'.format)
        kstable['cum_eventrate']=(kstable.events / data[target].sum()).cumsum()
        kstable['cum_noneventrate']=(kstable.nonevents / data['target0'].sum()).cumsum()
        kstable['KS'] = np.round(kstable['cum_eventrate']-kstable['cum_noneventrate'], 3) * 100

        #Formating
        kstable['cum_eventrate']= kstable['cum_eventrate'].apply('{0:.2%}'.format)
        kstable['cum_noneventrate']= kstable['cum_noneventrate'].apply('{0:.2%}'.format)
        kstable.index = range(1,len(kstable)+1)
        kstable.index.rename('Decile', inplace=True)
        pd.set_option('display.max_columns', 9)
        #Display KS
        #from colorama import Fore
        #print(Fore.RED + "KS is " + str(max(kstable['KS']))+"%"+ " at decile " + str((kstable.index[kstable['KS']==max(kstable['KS'])][0])))
        return(kstable)
    
    def calculate_event_and_volume_for_numerical(self,col,target):
        temp_ks = self.ks(self.df,target,col)
        max_ks_min_col_value = temp_ks.loc[temp_ks['KS'].idxmax(),'min_col']
        col_range = ['<='+str(max_ks_min_col_value),'>'+str(max_ks_min_col_value)]
        less_then_max_ks_min_col_val_event = sum(temp_ks[temp_ks['min_col']<=max_ks_min_col_value]['events'])
        greater_then_max_ks_min_col_val_event = sum(temp_ks[temp_ks['min_col']>max_ks_min_col_value]['events'])
        less_then_max_ks_min_col_val_nonevent = sum(temp_ks[temp_ks['min_col']<=max_ks_min_col_value]['nonevents'])
        greater_then_max_ks_min_col_val_nonevent = sum(temp_ks[temp_ks['min_col']>max_ks_min_col_value]['nonevents'])
        total_events = sum(temp_ks['events'])
        total_non_events = sum(temp_ks['nonevents'])

        event_rate = []
        event_rate.append(less_then_max_ks_min_col_val_event/total_events)
        event_rate.append(greater_then_max_ks_min_col_val_event/total_events)
        volume = []
        volume.append((less_then_max_ks_min_col_val_event+less_then_max_ks_min_col_val_nonevent)/(total_events+total_non_events))
        volume.append((greater_then_max_ks_min_col_val_event+greater_then_max_ks_min_col_val_nonevent)/(total_events+total_non_events))
        ks_temp = pd.DataFrame()
        
        ks_temp['categories'] = col_range
        ks_temp['feature'] = col
        ks_temp['event_rate'] = event_rate
        ks_temp['volume'] = volume
        ks_temp["event_ratio_volume"] = ks_temp['event_rate'] / ks_temp['volume']
        return ks_temp
        
    def __call__(self):
        return self.segmentation_df