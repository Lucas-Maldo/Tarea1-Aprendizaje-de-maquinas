# import tensorflow.keras.datasets as datasets
from tensorflow import keras
from keras import datasets
import matplotlib.pyplot as plt
import softmaxreg as softmaxreg
import numpy as np
from skimage.feature import hog
import metrics as metrics


def getSample(n_rows, n_cols, data):
    size = 28
    image = np.ones((n_rows*size, n_cols*size), dtype = np.uint8)*255
    n = n_rows * n_cols
    idx = np.random.randint(data.shape[0], size = n)
    
    i = 0
    for r in np.arange(n_rows) :
            for c in np.arange(n_cols) :
                image[r * size:(r + 1) * size, c * size : (c + 1) * size] = data[idx[i], : , : ]
                i = i + 1
    
    return image
    
if __name__ == '__main__' :
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

    #Inserting randomness into training data
    n = x_train.shape[0]
    idx = np.random.permutation(n)
    x_train = x_train[idx, :] 
    y_train = y_train[idx]

    #Inserting randomness into testing data
    n = x_test.shape[0]
    idx = np.random.permutation(n)
    x_test = x_test[idx, :] 
    y_test = y_test[idx]


    print ('{} {}'.format(x_train.shape, x_train.dtype))
    print ('{} {}'.format(x_test.shape, x_train.dtype))

    #Cambair formato de los datos
    x_train = x_train.reshape((60000,784))
    x_test = x_test.reshape((10000,784))

    #Normalizar
    mu = np.mean(x_train, axis = 0)
    dst = np.std(x_train, axis = 0)
    x_train = (x_train - mu) / dst
    x_test = (x_test - mu) / dst

    #Limpiar las divisiones por 0
    x_train[np.isnan(x_train)] = 0
    x_test[np.isnan(x_test)] = 0

    # digit = x_train[10,:];

    type = input("Usar entrada de la 'imagen' por si misma o usar la entrada 'hog': ")
    if (type == "hog"):
        x_train = x_train.reshape((60000,28,28))
        x_test = x_test.reshape((10000,28,28))
        digit = x_train[10,:,:];
        
        X_train = np.ones((x_train.shape[0],128))
        for i in range(x_train.shape[0]):
            X_train[i,:], hog_image1 = hog(x_train[i,:,:], orientations=8, pixels_per_cell=(7,7),
                            cells_per_block=(1, 1), visualize=True)
        X_test = np.ones((x_test.shape[0],128))
        for i in range(x_test.shape[0]):
            X_test[i,:], hog_image1 = hog(x_test[i,:,:], orientations=8, pixels_per_cell=(7,7),
                            cells_per_block=(1, 1), visualize=True)
        x_train = X_train
        x_test = X_test

    #Model
    SM = softmaxreg.SoftmaxReg(10)
    coeff = SM.fit(x_train, y_train)
    # print(coeff)

    y_pred = SM.predict(x_test)
    acc = metrics.multiclass_accuracy(y_test, y_pred)
    # y_test y acc
    # diccion = {0:[0,0], 1:[0,0], 2:[0,0], 3:[0,0], 4:[0,0], 5:[0,0], 6:[0,0], 7:[0,0], 8:[0,0], 9:[0,0]}
    # for i in range(x_test.shape[0]):
    #     if (acc[i] == 1):
    #         diccion[y_test[i]][0] += 1 #Valores correctos de esa clase
    #         diccion[y_test[i]][1] += 1 #Valores totales de esa clase
    #     if (acc[i] == 0):
    #         diccion[y_test[i]][1] += 1 #Valores totales de esa clase
    #Metricas

    confusion = metrics.confusion_matrix(y_test, y_pred, 10)
    valores = [0,1,2,3,4,5,6,7,8,9]
    promedios = []
    for i in range(len(confusion)):
        promedios.append(max(confusion[i])/sum(confusion[i]))
    # promedio = 0
    # for i in diccion.keys():
    #     diccion[i] = diccion[i][0]/diccion[i][1]
    #     promedio += diccion[i]
    # print(diccion)
    print(promedios)
    plt.title("Accuracy per class")
    plt.bar(valores, promedios)
    # plt.bar(diccion.keys(), diccion.values())
    print('Confusion matrix:\n {}'.format(confusion))
    print('acc {}'.format(np.mean(acc)))


    # print(fd)
    # fig, xs = plt.subplots(1,2)
    # xs[0].imshow(algo.reshape(28,28), cmap = 'gray')
    # xs[1].imshow(hog_image, cmap = 'gray')
    # print(fd.shape)
    # image = getSample(10,20, x_train)
    # print(image.shape)
    # plt.imshow(image, cmap = 'gray')
    # plt.axis('off')
    plt.show()
    
     