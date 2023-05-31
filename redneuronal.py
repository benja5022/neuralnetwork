import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Se cargan los datos de X
names1 = np.linspace(1,400,400)
dfX = pd.read_csv("X.csv", sep=",", names=names1)
X = np.matrix(dfX.values)


# Se cargan los datos de y
names2 = np.linspace(1,10,10)
dfY = pd.read_csv("Y.csv", sep=",", names=names2)
y = np.matrix(dfY.values)

class RedNeuronal():

    def __init__(self, nodes_per_layer, activation_functions=[]):
        self.nodes_per_layer = nodes_per_layer
        self.num_layers = len(nodes_per_layer) + 1
        self.activation_functions_names = activation_functions
        self.activation_functions = []
        self.weight = []
        self.bias = []

        for i, j in zip(self.weight, self.bias):
            print(j.shape)

        self.a_result = []
        self.zs = []
        self.deltas = []
        self.deltas_mayus = []

    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def __calculate_epsilon(self ,l_in, l_out):
        return np.sqrt(6.0)/np.sqrt(l_in + l_out)

    def __sigmoid(self,z):
        return 1 / (1 + np.exp(-z))
    
    def __sigmoid_prime(self, z):
        return np.multiply(self.__sigmoid(z),(1-self.__sigmoid(z)))
    
    def __relu(self, z):
        return np.maximum(0, z)
    
    def __relu_prime(self,z):
        return np.where(z > 0, 1, 0)
    
    def __get_mini_batches(self, X, y, mini_batch_size):

        n = X.shape[0]
        ms = mini_batch_size # mini batch size
        i_shuffled = np.random.permutation(X.shape[0])
        X_shuffled = X[i_shuffled]
        y_shuffled = y[i_shuffled]

        mini_batches = [ (X_shuffled[k:k+mini_batch_size],y_shuffled[k:k+mini_batch_size] ) for k in range(0, n, ms)]

        '''for i in range(n):
            if ( (i%ms == 0 and i != 0) or i == n-1):
                mini_batches.append( (X_shuffled[(i-ms): i] , y_shuffled[(i-ms): i]) )'''

        return mini_batches
    
    def __feedforward(self, X):
        activation = X

        for i in range(self.num_layers-1):
            self.zs.append( np.dot( activation, self.weight[i]) + self.bias[i])
            activation =  self.activation_functions[i][0]( self.zs[i]) #.__sigmoid( self.zs[i] )
            self.a_result.append(activation)
    
    def __backpropagation(self, y, x):

        nabla_b = [np.zeros(b.shape) for b in self.bias]
        nabla_w = [np.zeros(w.shape) for w in self.weight]

        
        delta = np.multiply((self.a_result[-1] - y), self.activation_functions[-1][1](self.zs[-1])) #.sigmoid_prime(self.zs[-1]) )
        #delta = self.a_result[-1] - y
        #print(self.activation_functions[-1])

        nabla_b[-1] = np.sum(delta, axis=0)
        nabla_w[-1] = np.dot(self.a_result[-2].T, delta)

        self.a_result.insert(0, x)

        for l in range(2,self.num_layers):
            z = self.zs[-l]
            #print(self.activation_functions[-l])
            delta =  np.multiply(np.dot(delta, self.weight[-l+1].T), self.activation_functions[-l][1](z) ) #.sigmoid_prime(z))
            nabla_b[-l] = np.sum(delta, axis=0)
            nabla_w[-l] = np.dot(delta.T, self.a_result[-l-1]).T
        
        return (nabla_w, nabla_b)

    def __update_weights(self, nabla_w,nabla_b, learning_rate, size):
        self.weight = [w - (learning_rate/size) * nb for w, nb in zip(self.weight, nabla_w)]
        self.bias = [b - (learning_rate/size) * nb for b, nb in zip(self.bias, nabla_b)]
    
    def __create_weights(self, X):
        
        for i in range(self.num_layers-1):
        
            if i == 0:
                epsilon = self.__calculate_epsilon(self.nodes_per_layer[i], X.shape[1])
                self.weight.append( (np.random.rand(self.nodes_per_layer[i], X.shape[1]) * 2.0 * epsilon - epsilon).T)
                self.bias.append(np.random.rand(1, self.nodes_per_layer[i]) )
            else:
                epsilon = self.__calculate_epsilon(self.nodes_per_layer[i], self.nodes_per_layer[i-1])
                self.weight.append( (np.random.rand(self.nodes_per_layer[i] , self.nodes_per_layer[i-1]) * 2.0 * epsilon - epsilon).T )
                self.bias.append(np.random.rand(1, self.nodes_per_layer[i]) )

    def asign_functions(self):
        if (self.num_layers-1) == len(self.activation_functions_names):
            for i in range(self.num_layers-1):
                if self.activation_functions_names[i] == 'sigmoid':
                    self.activation_functions.append((self.__sigmoid, self.__sigmoid_prime))

                if self.activation_functions_names[i] == 'relu':
                    self.activation_functions.append((self.__relu, self.__relu_prime))
        else:
            for i in range(self.num_layers-1):
                self.activation_functions.append((self.__sigmoid, self.__sigmoid_prime))




    def fit(self, x, y, iters, learning_rate, mini_batch_size):

        self.asign_functions()

        mini_batches = self.__get_mini_batches(x , y, mini_batch_size)
        J = []
        self.__create_weights(x)
        for i in range(iters):
            for mini_batch in mini_batches:
                self.__feedforward(mini_batch[0])
                nabla_w,nabla_b = self.__backpropagation(mini_batch[1], mini_batch[0] )
                self.__update_weights(nabla_w,nabla_b, learning_rate, mini_batch[0].shape[0])
                J.append(self.__get_cost(mini_batch[0],mini_batch[1]))

                self.a_result=[]
                self.zs =[]
            #learning_rate = learning_rate * 0.999

        plt.plot(np.linspace(0,len(J),len(J)), J)
        plt.show()
        
        
            
    def __get_cost(self, X, y):
        sum_regular = 0
        m = X.shape[0]
        for i in range(self.num_layers-1):
            sum_regular = sum_regular + np.sum(np.sum( np.multiply(self.weight[i][:,1:], self.weight[i][:,1:])))

        num = len(self.a_result) -1
        J = (1/(m*2)) * np.sum(np.sum(- np.multiply(y, np.log(self.a_result[num])) - np.multiply((1-y),np.log(1 - self.a_result[num])))) + (1 / (2*m)) * sum_regular

        return J
            
    def predict(self,X):
        a = X
        for i in range(self.num_layers-1):
            a = self.activation_functions[i][0]( a * self.weight[i] + self.bias[i])  #__sigmoid( a * self.weight[i] + self.bias[i])
        return np.round(a)

red = RedNeuronal([40,10], [ 'relu', 'sigmoid'])

red.fit(X, y, 50, 1.5, 500) # X, y , 500, 3, 250

fium = red.predict(X)

cont = 0
for i in range(fium.shape[0]):
    if( (fium[i] == (y[i])).all() ):
        cont = cont + 1
print(cont)

'''
X_frutas = np.matrix([ [1.0,1.0,0.0,1.0], [1.0,-1.0,0.0,1.0], [-1.0,1.0,1.0,-1.0], [0.0,1.0,-1.0,0.0]])
y_frutas = np.matrix([ [0.0,0.0],[1.0,1.0],[1.0,0.0],[0.0,1.0]] )
print(X_frutas.shape)
print(y_frutas.shape)

red_frutas = RedNeuronal([20,2])

red_frutas.fit(X_frutas,y_frutas, 1000, 0.1, 4)
cont = 0
result = red_frutas.predict(X_frutas)

for i in range(result.shape[0]):
    if( (result[i] == (y_frutas[i])).all() ):
        cont = cont + 1
print(cont)
'''