# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 16:19:12 2021

@author: ali hussain
"""

f=open("D:/mystuff/HealthCareProject.csv")
lines=f.readlines()[2:]
print(lines)

x=[]
y=[]

for line in lines:
    w=line.strip().split(",")
    ins=w[1:13]
    out=w[13]
    x.append(ins)
    y.append(out)
    
    
print(x)
print(y)  

import numpy as np
ins=np.array(x)

print(ins)

out=np.array(y)
print(out)



def clean(x):
    n=[]
    for v in x:
        c=[float(v) for v in x if v!='999']
        m=sum(c)/len(c)
        if v=='999':
            noise=np.random.randn()/1000
            n.append(m+noise)
        else:
            n.append(float(v))
    return n     

z1=ins[0:,0:1]
print(z1)

v1=clean(z1)
print(v1)

v3=clean(ins[0:,2:3])
print(v3)

v9=clean(ins[0:,8:9])
print(v9)

v10=clean(ins[0:,9:10])
print(v10)

v11=clean(ins[0:,10:11])
print(v11)

def cleanc(x):
    count1=0
    count2=0
    n=[]
    for v in x:
        if v=='1':
            n.append(1)
            count1+=1
        elif v=='2':
            n.append(2)
            count2+=1
        else:
            if count1>count2:
                n.append(1)
            else:
                n.append(2)
            
        
    return n        
        
f1=ins[0:,1:2]
print(f1)

v2=cleanc(f1)
print(v2)

v5=cleanc(ins[0:,4:5])
print(v5)

v7=cleanc(ins[0:,6:7])
print(v7)

v12_1=cleanc(ins[0:,11:12])
v12=[float(v) for v in v12_1]

def clean3(x):
    count1=0
    count2=0
    count3=0
    n=[]
    for v in x:
        if v=='1':
            n.append(1)
            count1+=1
        elif v=='2':
            n.append(2)
            count2+=1
        elif v=='3':
            n.append(3)
            count3+1
        else:
            if count1>count2 & count1>count3:
                n.append(1)
            elif count2>count1 & count2>count3:
                n.append(2)
            else:
                n.append(3)
            
        
    return n       


v4=clean3(ins[0:,3:4])
print(v4)

v6=clean3(ins[0:,5:6])
print(v6)

def clean4(x):
    count1=0
    count2=0
    count3=0
    count4=0
    n=[]
    for v in x:
        if v=='1':
            n.append(1)
            count1+=1
        elif v=='2':
            n.append(2)
            count2+=1
        elif v=='3':
            n.append(3)
            count3+1
        elif v=='4':
            n.append(4)
        
        else:
            if count1>count2 & count1>count3 & count1>count4:
                n.append(0)
            elif count2>count1 & count2>count3 & count2>count4:
                n.append(1)
            elif count3>count1 & count3>count2 & count3>count4:
                n.append(2)
            else:
                n.append(3)
            
        
    return n      


v8=clean4(ins[0:,7:8])
print(v8)

print(v1)

print(v2)

print(v3)

print(v4)

print(v5)

print(v6)

print(v7)

print(v8)

print(v9)

print(v10)

print(v11)

print(v12)


X=np.c_[v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12]
print(X)


Y=[int(v) for v in out]

print(Y)



from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
model=GaussianNB()
model.fit(X,y)
ycap=model.predict(X)
#print(ycap)
accuracy_score(y,ycap)
print(np.c_[Y,ycap])


# 0.19313304721030042



from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
model=DecisionTreeClassifier()
model.fit(X,y)
ycap=model.predict(X)
#print(ycap)
print(np.c_[Y,ycap])
accuracy_score(y,ycap)


#0.9656652360515021


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
model=RandomForestClassifier()
model.fit(X,y)
ycap=model.predict(X)
#print(ycap)
print(np.c_[Y,ycap])
accuracy_score(y,ycap)


#  0.944206008583691



from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
model=SVC()
model.fit(X,y)
ycap=model.predict(X)
#print(ycap)
print(np.c_[Y,ycap])
accuracy_score(y,ycap)


#0.8540772532188842


