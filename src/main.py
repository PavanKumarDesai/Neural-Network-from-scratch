import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class NN:

    ''' X and Y are dataframes '''
    def __init__(self,layer_dims):
        self.layer_dims=layer_dims
        
    def initialize_parameters_deep(self,layer_dims):
        np.random.seed(3)
        parameters = {}
        L = len(layers_dims) - 1 # number of layers
        for l in range(1, L + 1):
            parameters['W' + str(l)] = (np.random.randn(layers_dims[l],layers_dims[l-1]))*(np.sqrt(2./layers_dims[l-1]))
            parameters['b' + str(l)] = np.zeros((layers_dims[l],1))
            assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
            assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
        return parameters
    
    def sigmoid(self,Z):
        A = 1/(1+np.exp(-Z))
        cache = Z
        return A, cache

    def relu(self,Z):
        A = np.maximum(0,Z)
        assert(A.shape == Z.shape)
        cache = Z 
        return A, cache


    def relu_backward(self,dA, cache):
        Z = cache
        dZ = np.array(dA, copy=True) 
        dZ[Z <= 0] = 0
        assert (dZ.shape == Z.shape)
        return dZ

    def sigmoid_backward(self,dA, cache):
        Z = cache
        s = 1/(1+np.exp(-Z))
        dZ = dA * s * (1-s)
        assert (dZ.shape == Z.shape)
        return dZ
    
    def linear_forward(self,A, W, b):
        Z = np.dot(W,A)+b
        assert(Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)
        return Z, cache

    def linear_activation_forward(self,A_prev, W, b, activation):
        if activation == "sigmoid":
            Z, linear_cache = self.linear_forward(A_prev,W,b)
            A, activation_cache = self.sigmoid(Z)
    
        elif activation == "relu":
            Z, linear_cache = self.linear_forward(A_prev,W,b)
            A, activation_cache = self.relu(Z)
    
        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)

        return A, cache
    
    def L_model_forward(self,X, parameters):
        caches = []
        A = X
        L = len(parameters) // 2                  # number of layers 
        for l in range(1, L):
            A_prev = A 
            A, cache = self.linear_activation_forward(A_prev, 
                                                parameters["W" + str(l)], 
                                                parameters["b" + str(l)], 
                                                activation='relu')
            caches.append(cache)
        AL, cache = self.linear_activation_forward(A, 
                                                parameters["W" + str(L)], 
                                                parameters["b" + str(L)], 
                                                activation='sigmoid')
        caches.append(cache)
        assert(AL.shape == (1,X.shape[1]))
            
        return AL, caches
    def compute_cost(self,AL, Y):
        m = Y.shape[1]
        cost = -1/m*np.sum(np.multiply(Y,np.log(AL))+np.multiply((1-Y),np.log(1-AL)))
        cost = np.squeeze(cost)      
        assert(cost.shape == ())
        return cost
    def linear_backward(self,dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]
        dW = 1/m*np.dot(dZ,A_prev.T)
        db = 1/m*np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T,dZ)
        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)
        return dA_prev, dW, db

    def linear_activation_backward(self,dA, cache, activation):
        linear_cache, activation_cache = cache
    
        if activation == "relu":
            dZ = self.relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ,linear_cache)
        elif activation == "sigmoid":
            dZ = self.sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ,linear_cache)
        return dA_prev, dW, db
    def L_model_backward(self,AL, Y, caches):
        grads = {}
        L = len(caches) # the number of layers
        m = AL.shape[1]
        Y = Y.reshape(AL.shape) 
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        current_cache = caches[-1]
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(dAL,current_cache,activation = "sigmoid")
        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l + 1)],current_cache,activation = "relu")
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp
        return grads
    def update_parameters(self,parameters, grads, learning_rate):
        L = len(parameters) // 2 
        for l in range(L):
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        return parameters
    
    def L_layer_model(self,X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
        np.random.seed(1)
        costs = []                         # keep track of cost
        parameters = self.initialize_parameters_deep(layers_dims)
        for i in range(0, num_iterations):
            AL, caches = self.L_model_forward(X, parameters)
            cost = self.compute_cost(AL, Y)
            grads = self.L_model_backward(AL, Y, caches)
            parameters = self.update_parameters(parameters, grads, learning_rate)
            
            if print_cost and i % 100 == 0:
                costs.append(cost)
            
        return parameters
    def predict(self,X, y, parameters):
        m = X.shape[1]
        n = len(parameters) // 2 
        p = np.zeros((1,m))
    
        # Forward propagation
        probas, caches = self.L_model_forward(X, parameters)

    
        # convert probas to 0/1 predictions
        for i in range(0, probas.shape[1]):
            if probas[0,i] > 0.6:
                p[0,i] = 1
            else:
                p[0,i] = 0
    
        #print results
        #print ("predictions: " + str(p))
        #print ("true labels: " + str(y))
        #print("Accuracy: "  + str(np.sum((p == y)/m)))
        acc=(np.sum((p == y)/m))
        
        return p,acc

    def CM(self,y_test,y_test_obs):
        '''
        Prints confusion matrix 
        y_test is list of y values in the test dataset
        y_test_obs is list of y values predicted by the model

        '''

        for i in range(len(y_test_obs)):
            if(y_test_obs[i]>0.6):
                y_test_obs[i]=1
            else:
                y_test_obs[i]=0
        
        cm=[[0,0],[0,0]]
        fp=0
        fn=0
        tp=0
        tn=0
        
        for i in range(len(y_test)):
            if(y_test[i]==1 and y_test_obs[i]==1):
                tp=tp+1
            if(y_test[i]==0 and y_test_obs[i]==0):
                tn=tn+1
            if(y_test[i]==1 and y_test_obs[i]==0):
                fp=fp+1
            if(y_test[i]==0 and y_test_obs[i]==1):
                fn=fn+1
        cm[0][0]=tn
        cm[0][1]=fp
        cm[1][0]=fn
        cm[1][1]=tp
        acc2=(tp+tn)/(tp+tn+fp+fn)
        p= tp/(tp+fp)
        r=tp/(tp+fn)
        f1=(2*p*r)/(p+r)
        #print()
        print("Confusion Matrix : ")
        print(cm)
        print()
        #print(f"Accuracy : {acc2}")
        print(f"Precision : {p}")
        print(f"Recall : {r}")
        print(f"F1 SCORE : {f1}")
            


    

if __name__=="__main__":
    df=pd.read_csv("../data/pre_processed.csv")
    #df=pre_processing(df)
    df['Age']=pd.to_numeric(df['Age'])
    df['Weight']=pd.to_numeric(df['Weight'])
    df['HB']=pd.to_numeric(df['HB'])
    df = df.astype({"Community_1": np.uint8})
    df = df.astype({"Community_2": np.uint8})
    df = df.astype({"Community_3": np.uint8})
    df = df.astype({"Community_4": np.uint8})
    nrow=len(df.index)
    ncol=len(df.columns)
    X=df.iloc[:,0:(ncol-1)].values.reshape(nrow,(ncol-1))
    Y=df.iloc[:,(ncol-1)].values.reshape(nrow,1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
    X_train=X_train.transpose()
    X_test=X_test.transpose()
    Y_train=Y_train.transpose()
    Y_test=Y_test.transpose()
    input_layer_size=X_train.shape[0]
    layers_dims=[input_layer_size,15,5,2,1]
    param=NN(layers_dims)
    para=param.L_layer_model(X_train, Y_train,layers_dims,learning_rate=0.01, num_iterations = 9000)
    pred_train,acc1 = param.predict(X_train, Y_train, para)
    pred_test,acc2 = param.predict(X_test,Y_test, para)
    print("Training Accuracy: ",acc1)
    print("Test Accuracy: ",acc2)
    check1=pred_test.tolist()
    check2=Y_test.tolist()
    print()
    print("For Test data:")
    param.CM(check2[0],check1[0])

