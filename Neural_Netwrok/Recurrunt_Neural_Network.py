import numpy as np
from .Activition_functions import Activition_functions



class RNN():
    def __init__(self, word: str, k: int, d: int, W_H: np.array = None, W_X: np.array = None, W_Y: np.array = None):
        self.k = k  
        self.d = d  
        self.word = word
        
        self.W_H = np.random.uniform(-1, 1, size=(self.d, self.d)) if W_H is None else W_H
        self.W_X = np.random.uniform(-1, 1, size=(self.d, self.k)) if W_X is None else W_X
        self.W_Y = np.random.uniform(-1, 1, size=(self.k, self.d)) if W_Y is None else W_Y
       
        self.h = {"t0": np.zeros((self.d, 1))}  
        self.letter = self.__one_hot_list_Char() if len(self.word.split(" ")) == 1 else self.__one_hot_list_word()
        self.cache = {}  
        self.loss_history = []  

    def __one_hot_list_Char(self):
        unique_chars = sorted(list(set(self.word)))
        char_to_index = {char: idx for idx, char in enumerate(unique_chars)}
        one_hot_vectors = []
        for char in self.word:
            vector = np.zeros(len(unique_chars))
            vector[char_to_index[char]] = 1
            one_hot_vectors.append(vector)
        return one_hot_vectors

    def __one_hot_list_word(self):
        unique_words = self.word.split(" ")
        word_to_index = {word: idx for idx, word in enumerate(unique_words)}
        one_hot_vectors = []
        for word in unique_words:
            vector = np.zeros(len(unique_words))
            vector[word_to_index[word]] = 1
            one_hot_vectors.append(vector)
        return one_hot_vectors

    def forward(self):
        self.cache = {}
        self.h = {"t0": np.zeros((self.d, 1))}  
        for t in range(len(self.letter) - 1):
            x_t = np.array(self.letter[t]).reshape((-1, 1)) 
            h_prev = self.h[f"t{t}"] 
            z_t = np.dot(self.W_H, h_prev) + np.dot(self.W_X, x_t)  
            h_t = np.tanh(z_t)  
            self.cache[f"t{t+1}"] = (x_t, h_prev, z_t)
            self.h[f"t{t+1}"] = h_t
        

        y_pred = self.W_Y.dot(self.h[f"t{len(self.letter)-1}"])  
        y_pred_softmax = Activition_functions.softmax(y_pred)
        loss = self.mean_squared_error(y_pred_softmax, np.array(self.letter[-1]).reshape((-1, 1)))
        return y_pred_softmax, loss

    def backward(self, learning_rate=0.01):
        T = len(self.letter) - 1
        dW_H = np.zeros_like(self.W_H)
        dW_X = np.zeros_like(self.W_X)
        dW_Y = np.zeros_like(self.W_Y)
        dh_next = np.zeros_like(self.h["t0"])
        
        
        y_true = np.array(self.letter[-1]).reshape((-1, 1))
        y_pred, _ = self.forward()
        dy_pred = y_pred - y_true  
        
        
        dW_Y = dy_pred.dot(self.h[f"t{T}"].T)  
        
        
        for t in reversed(range(1, T + 1)):
            x_t, h_prev, z_t = self.cache[f"t{t}"]
            dh_t = self.W_Y.T.dot(dy_pred) + dh_next 
            dz_t = dh_t * (1 - np.tanh(z_t) ** 2)  
            dW_H += dz_t.dot(h_prev.T)
            dW_X += dz_t.dot(x_t.T)
            dh_next = self.W_H.T.dot(dz_t)
        
        
        self.W_H -= learning_rate * dW_H
        self.W_X -= learning_rate * dW_X
        self.W_Y -= learning_rate * dW_Y
        return dW_H, dW_X, dW_Y

    def train(self, epochs=100, learning_rate=0.01, verbose=True):
        for epoch in range(epochs):
            prediction, loss = self.forward()
            self.backward(learning_rate)
            self.loss_history.append(loss)
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch}, Loss: {np.mean(loss):.4f}")
        return prediction
    

    def mean_squared_error(self,y_pred, y_true):
        shape = np.array(y_pred).shape
        return  (np.array(y_pred) - np.array(y_true).reshape(shape))**2 