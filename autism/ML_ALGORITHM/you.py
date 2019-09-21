
def getPrediction(x):
    from sklearn import tree
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    import numpy 
    dataset=numpy.loadtxt("C:/Users/keerthivasan/Desktop/Sairam_Hackathon/sairam_hackathon/autism/ML_ALGORITHM/prima-indians-autism.csv", delimiter= ",")
    X=dataset[:,0:8]
    Y=dataset[:,8]
    model=Sequential()
    model.add(Dense(15, input_dim=8, activation="relu"))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(8, activation="relu"))
    model.add(Dropout(.2))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.fit(X,Y, epochs=200, batch_size=10)
   
    list = []
    list.append(x)
    numpy.savez('geekfile.npz', a = list) 
    
    # compressed file is loaded 
    c = numpy.load('geekfile.npz') 
    classes= model.predict(c['a'],batch_size=1)
    c=classes*100
    return c[0]




                    
#numerical gradient approxiamation
#slope=gradient
#1/1+e^-x sigmoid
#sigmod activation
#logistic regression

