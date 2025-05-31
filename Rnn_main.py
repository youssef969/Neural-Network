from Neural_Netwrok.Recurrunt_Neural_Network import RNN

rnn = RNN(word="I am Youssef Mustafa", k=4, d=3) 
y_pred, _ = rnn.forward()
print("Final prediction after Forward pass:", y_pred)

final_output = rnn.train(epochs=200, learning_rate=0.01)
print("Final prediction after Backpropagation pass:",final_output)
