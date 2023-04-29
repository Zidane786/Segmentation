from sklearn.tree import DecisionTreeClassifier


class DecisionTreeSegmentation:
    def __init__(self,parameters):
        self.parameters = self.get_param(parameters)

    def get_param(self,parameter):
        key_vals = parameter.split(',')
        parameter_dict =  {key_val.split(':')[0]:key_val.split(':')[-1] for key_val in key_vals }
        return parameter_dict
    
    