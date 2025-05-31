import random as rn
import numpy as np 
from .Activition_functions import Activition_functions

class ANN():
    def __init__(self,input_layer:int,Hidden_layers:list,output_layer:int,activition_function:Activition_functions,alpha:float):
        self.output_layer = output_layer
        self.input_layer = input_layer
        self.Hidden_layers = Hidden_layers
        self.activition_function = activition_function
        self.alpha = alpha
        self.index = 0
        self.derivitive_activition_function =  self.__Backpropigation_activition_function()
        

    def __Backpropigation_activition_function(self):
        active_func_name =  self.activition_function.__name__
        if  active_func_name == "sigmoid":
            return Activition_functions.sigmoid_derivative

    def __Hidden_layer_nerouns(self):
        if self.index < len(self.Hidden_layers):
            element = self.Hidden_layers[self.index]
            self.index += 1
            return element
        else:
            return None  

    def Input_layer(self,input_layer:dict | None ) -> dict :
        if input_layer == None:
            input_layer = {f"i{i}": rn.uniform(-0.5,0.5) for i in range(self.input_layer)}
            print(f"Intial Values to Input Layer = {input_layer}")
            return input_layer
        else:
            print(f"Intial Values to Input Layer = {input_layer}")
            return input_layer
    
        
    
    def Hidden_layer(self,weghits:list,input_layer:dict,bais:float):
        """weights here is Weights from input Neuron to Next Neuron"""

        Nerons = self.__Hidden_layer_nerouns()
        hidden_layer = {f"h{i}": 0 for i in range(Nerons)}
        pathes = Nerons * len(input_layer)
        inputs = np.array(list(input_layer.values()))
        
        if pathes != len(weghits):
            raise ValueError("Logical Error: Number of pathes not uqual number of weights")
        elif Nerons == None:
            raise ValueError("Logical Error: Error")
        else: 
            inputs = inputs.reshape(1,len(input_layer))
            weights_matrix = np.array(weghits).reshape(len(input_layer),Nerons)
            result = np.dot(inputs ,weights_matrix)
            result = result + bais 
            for index , Value in zip([x for x in range(Nerons)],result[0].tolist()):
                hidden_layer[f"h{index}"] = self.activition_function(Value)
        print(f"Hidden Layer: {Nerons} Nerouns Values = {hidden_layer}")
        return hidden_layer

        
    def Output_layer(self,weghits:list,Hidden_layer:dict,bais:float) -> dict:
        """weights here is Weights from input Neuron to Next Neuron"""

        output_layer = {f"o{i}": 0 for i in range(self.output_layer)}
        pathes = len(Hidden_layer) * self.output_layer
        if pathes != len(weghits):
            raise ValueError("Logical Error: Number of pathes not uqual number of weights")
        else:
            inputs = np.array(list(Hidden_layer.values())).reshape(1,len(Hidden_layer))
            weights_matrix = np.array(weghits).reshape(len(Hidden_layer),len(output_layer))      
            result = np.dot(inputs,weights_matrix,)
            result = result + bais
            for index , Value in zip([x for x in range(self.output_layer)],result[0].tolist()):
                output_layer[f"o{index}"] = self.activition_function(Value)
        print(f"Output Layer Nerouns Values = {output_layer}")
        return output_layer
    
    def Total_error(self,targets:list,output_layer:dict):
        error = 0
        for target , output in zip(targets,list(output_layer.values())):
           error = error  +  0.5*(target - output)**2
        print(f"Error Rate = {error * 100} %")
        return error
    
    def Backpropagation(self, input_layer:dict, hidden_layer:dict, output_layer:dict, weights_hidden:list, weights_output:list, targets:list):
        input_layer = np.array(list(input_layer.values()))
        hidden_layer = np.array(list(hidden_layer.values()))
        output_layer = np.array(list(output_layer.values())) 
        targets = np.array(targets)
        
        #! output Layer update Weight
        alpha_output = -1*(targets - output_layer) * self.derivitive_activition_function(output_layer)
        delta_output =alpha_output * hidden_layer * self.alpha
        output_updated_weights = np.array(weights_output).reshape(2,self.output_layer) - delta_output
        output_updated_weights_list = list(output_updated_weights.flatten())
        print([f"{y} Updated to: {x}" for x,y in zip(output_updated_weights_list , weights_output)])
        

        #! Hidden Layer update Weight
        alpha_hidden_layer = np.dot(alpha_output,np.array(weights_output).reshape(2,2)) * self.derivitive_activition_function(hidden_layer) 
        delta_hideen_layer = alpha_hidden_layer * self.alpha * input_layer
        hidden_updated_weights = np.array(weights_hidden).reshape(2,self.output_layer) - delta_hideen_layer
        hidden_updated_weights_list = list(hidden_updated_weights.flatten())
        print([f"{y} Updated to: {x}" for x,y in zip(hidden_updated_weights_list, weights_hidden)])
