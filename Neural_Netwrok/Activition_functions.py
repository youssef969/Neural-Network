import math
import numpy as np
from math import exp

class Activition_functions():

    @staticmethod
    def Tanh(z):
        return (math.exp(z) - math.exp(-z)) / (math.exp(z) + math.exp(-z))
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))
    
    @staticmethod
    def Relu(value):
        return max(0,value)
    
    @staticmethod
    def Linear(value):
        return value
    
    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)
    
    @staticmethod
    def softmax(vector):
        e = np.exp(vector)
        return e / e.sum()
    
    def __Softmax(array:list):
        summation = sum([exp(x) for x in array])
        x = map(lambda x: round(exp(x)/ summation,2),array)
        return list(x)
