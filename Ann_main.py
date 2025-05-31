from Neural_Netwrok.Artificial_Neural_Network import ANN
from Neural_Netwrok.Activition_functions import Activition_functions

nn = ANN(input_layer=2,Hidden_layers=[2],output_layer=2,activition_function=Activition_functions.sigmoid,alpha=0.6)


input_layer = nn.Input_layer(input_layer={"i1":0.1,"i2":0.5})
hidden_layer = nn.Hidden_layer(input_layer=input_layer,weghits=[0.1,0.2,0.3,0.4],bais=0.25)
output_layer = nn.Output_layer(weghits=[0.5,0.7,0.6,0.8],Hidden_layer=hidden_layer,bais=0.35,)
total_error = nn.Total_error(targets=[0.05,0.05],output_layer=output_layer)

b = nn.Backpropagation(
    input_layer=input_layer,
    hidden_layer=hidden_layer,
    output_layer=output_layer,
    weights_hidden=[0.1,0.2,0.3,0.4],
    weights_output=[0.5,0.7,0.6,0.8],
    targets=[0.05,0.05]
)