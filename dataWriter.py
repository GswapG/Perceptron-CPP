import pickle
import gzip
import numpy as np
import matplotlib.pyplot as mp
import pyttsx3
path = './data/mnist.pkl.gz'

# engine = pyttsx3.init()
with gzip.open(path, 'rb') as f:
    obj1 = pickle.load(f, encoding = 'unicode-escape')
    training_data, test_data, validation_data = obj1

## Data is of the form : 
# [
#     [ Input 
#         [784...],[784...],[784...]
#     ]
#     ,
#     [ Output
#         int,int,int,....
#     ]
# ]

## Converting to the form : 
# [
#     [[input(784)],output(10)]
#     ,
#     [[input(784)],output(10)]
#     ,
#     ...
# ]

## Output should be a vector of size 10 
## where value of out[i] = 1 when digit is i

with open("training_data.txt" , "w") as f:
    for i in range(0, len(training_data[0])):
        for j in range(0,784):
            temp1 = training_data[0][i][j]
            f.write(f"{temp1:.8f}")
            f.write(" ")
        temp1 = training_data[1][i]
        f.write(f"{temp1}")
        f.write("\n")

with open("test_data.txt" , "w") as f:
    for i in range(0, len(test_data[0])):
        for j in range(0,784):
            temp1 = test_data[0][i][j]
            f.write(f"{temp1:.8f}")
            f.write(" ")
        temp1 = test_data[1][i]
        f.write(f"{temp1}")
        f.write("\n")

with open("validation_data.txt" , "w") as f:
    for i in range(0, len(validation_data[0])):
        for j in range(0,784):
            temp1 = validation_data[0][i][j]
            f.write(f"{temp1:.8f}")
            f.write(" ")
        temp1 = validation_data[1][i]
        f.write(f"{temp1}")
        f.write("\n")

# mp.show()

# def speakNum(number):
#     txt = str(number)
#     engine.say(txt)
#     engine.runAndWait()

# def display(k):
#     grid = []

#     for i in range(0,28):
#         temp = []
#         for j in range(0,28):
#             temp.append(255*float(training_data[0][k][i*28 + j]))
#         grid.append(temp)

#     mp.imshow(grid, cmap = "gray", vmin = 0, vmax = 255)
#     mp.draw()
#     mp.pause(1)
#     speakNum(int(training_data[1][k]))
#     mp.pause(1)

# for k in range(0,100):
#     display(k)
