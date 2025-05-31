import numpy as np 



class CNN():

    def __init__(self,Image_channel:np.ndarray, filter: np.ndarray,Pooling_size=2, mode="max", stride: int = 1):
        self.Image_channel = Image_channel
        self.filter =filter
        self.Pooling_size =Pooling_size
        self.mode =mode
        self.stride = stride
        

        self.Filerting = self.CNN_func(self.Image_channel,self.filter,self.stride)
        

    
    def CNN_func(self,input: np.ndarray, filter: np.ndarray, stride: int = 1):
        input_size = input.shape[0]
        filter_size = filter.shape[0]
        output_size = int(((input_size - filter_size) / stride) + 1)
        output = np.zeros((output_size, output_size))
        
        for i in range(0, output_size * stride, stride):
            for j in range(0, output_size * stride, stride):
                region = input[i:i+filter_size, j:j+filter_size]
                output[i // stride, j // stride] = np.sum(region * filter, dtype=np.int16)
                
                sum_values = []
                for row in range(filter_size):
                    for col in range(filter_size):
                        sum_values.append(f"({region[row, col]} x {filter[row, col]})")
                
                sum_expression = " + ".join(sum_values)
                print(f"{sum_expression} = {output[i // stride, j // stride]}")
                print("\n")
        
        return output
    

    def get_value(self):
        return self.Filerting