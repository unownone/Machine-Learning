import numpy as np
import pandas as pd
import json
import os 
import random

#sigmoid function
def sigmoid(z):
    z = z.astype(float)
    s=1/(1+np.exp(-z))
    return s

#prediction to use the algorithm
def predict(w,b,X):

    m=X.shape[1]
    Y_predict=np.zeros((1,m))
    w=w.reshape(X.shape[0],1)

    #reshaping and initializing the values
    A=sigmoid(np.dot(w.T,X)+b)

    #vectorized implementation : 

    return A


print("Welcome! Do you wanna know if you would survive if you boarded the Titanic: The UNSINKABLE SHIP?")

name=input("Please input your name ::")

age=float(input("Please input your age ::"))

sex=1.0 if input("Please input your sex ::")=="male" else 0.0

sibl=float(input("How many siblings do you wanna have with you? ::"))

parents=float(input("How many parents do you wanna have with you? ::"))

fare=float(input("How much are you willing to pay? ::"))

P_class=random.randint(1,3)

send_data=np.asarray([P_class,sex,age,sibl,parents,fare])
status=os.stat("learning_parameter.json").st_size==0
data=0
if not status:
    data=json.load(open("learning_parameter.json","r"))
else:
    print("Please run the ipynb to train the model before trying to use it")
    
w=np.asarray(data["w"])
b=data["b"]
send_data=send_data.reshape(w.shape[0],w.shape[1])
send_data=send_data/send_data.std()
#starting the process of prediction
prediction=predict(w,b,send_data)
p=(prediction[0]>=0.5)
print("Hi")
print("Mr",name) if sex==1 else print("Miss/Mrs",name)
print("We sent back one of your clones back in 1912 to board the titanic.")
print("And it turns out you are going to -",("Survive" if p else "Not Survive"),"the Titanic's ride!")
print("Your survival rate is : ",prediction)
if p:
    print("YES! YOU ARE ALIVE. Go live your life or something idc...")
else:
    print("Ahhh!Sorry to see you go like that.")
    print("That's why I told you not to do it : ). Now pay the fee")
