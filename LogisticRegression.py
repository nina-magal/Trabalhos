
import numpy as np

def sigmoid(s):
    return 1/(1+np.exp(-s))

class LogisticRegression():
    #primeiro parametrizamos o learning rate, o número de iterações, os pesos e os bias. Todos começam com um valor padrão.
    def __init__(self, lr = 0.001, n_iters= 1000):
        self.lr = lr 
        self.n_iters = n_iters 
        self.weights = None #iniciamos os pesos como 0
        self.bias = None #iniciamos os bias como 0
        
    #agora criamos o modelo de regressão logística
    def fit(self, X, y ):

        #definimos o formato da amostra de entrada do modelo
        n_samples , n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        #minimizando a função de perda
        for _ in range(self.n_iters):
            linear = np.dot(X, self.weights) + self.bias #y = β0 + β1x1 + ... +βnxn
            predictions = sigmoid(linear) #y' =  1/1+e**-(β0 + β1x1 + ... +βnxn)

            #calcular o gradiente da entropia cruzada
            dw = (1/n_samples)* 2*(np.dot(X.T,(predictions-y))) #derivada em relação ao peso
            db = (1/n_samples)* 2*(np.sum(predictions-y)) #derivada em relação ao bias

            #encontrando os parâmetros que minimizam a função de perda
            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db
    #criando classificador
    def predict(self,X):
        
        linear = np.dot(X,self.weights) + self.bias
        y_pred = sigmoid(linear)
        class_pred = [0 if y <=0.5 else 1 for y in y_pred]
        return class_pred
    
