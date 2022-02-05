import pandas as pd 
import  numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df= pd.read_csv("E:\\py\\Data\\data.csv")

def data_description():
    k=input("chose what u want to know about the data (columns,corrlition,info,isnull,show,describe)\n")
    if k == 'columns':
       print (df.columns)
  
    elif k =='corrlition':
        print (df.corr())
       
    elif k =='info':
        print (df.info())
        
    elif k =='isnull':
        print (df.isnull().sum())
        
    elif k =='show':
        print (df.head(20))
        
    elif k =='describe':
        print (df.describe())
    else:
        print("sorry the opration u chosie is not supported :(")     
df.index=df['id']
df.drop(['id','Unnamed: 32'],axis=1,inplace=True)

X=df.iloc[:,1:]
y = df.iloc[:,:1]
from sklearn.preprocessing import LabelBinarizer
binarize=LabelBinarizer()
y=binarize.fit_transform(y)


corrlition=X.corrwith(df['diagnosis']=="M")
corrlition.plot(kind='bar',color='blue',xlabel="Medical analysis",ylabel="Illness",title="corlation bettween Medical analysis and illness")
#removeing  non corlational data
low_corrlation_Features=corrlition[corrlition<.2].index
X.drop(low_corrlation_Features,axis=1)

#Visulizing data with Heatmap
predictors=df.columns[1:]
plt.figure(figsize=(18,18))
sns.heatmap(df[predictors].corr(),linewidths = 1, annot = True, fmt = ".2f")
plt.show()

#spliting the data for traing and testing 

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42,shuffle=True)

#print(y_test.shape)

#using SVC_clasfiction
from sklearn.svm import SVC
svc_clasfiction=SVC(kernel="linear",C=.5)
svc_clasfiction.fit(X_train, y_train)
y_pred=svc_clasfiction.predict(X_test)
print("\n Accuracy for SVC :",svc_clasfiction.score(X_test, y_test)*100,"% \n")
from sklearn.metrics import classification_report
print("classfing using SVC classification_report \n",classification_report(y_test, y_pred))
#using knn_clasfiction
from sklearn.neighbors import KNeighborsClassifier
#testing which is the best n_neighbors

error1=[]
error2=[]
for k in range(1,15):
    knn= KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred1= knn.predict(X_train)
    error1.append(np.mean(y_pred1))
    y_pred2= knn.predict(X_test)
    error2.append(np.mean(y_pred2))
    
plt.figure('test K')
plt.plot(range(1,15),error2,label="test")
plt.plot(range(1,15),error1,label="train")
plt.xlabel('k Value')
plt.ylabel('Error')
plt.legend()
plt.show()
# the lest amount of erorr for test and train is 11
knn_clasfiction=KNeighborsClassifier(n_neighbors=11)
knn_clasfiction.fit(X_train,y_train)
y_pred2=knn_clasfiction.predict(X_test)
print("Accuracy for KNN :",knn_clasfiction.score(X_test, y_test)*100,"% \n")
print("classfing using KNN classification_report \n",classification_report(y_test, y_pred2))
print("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\00\n")
y_test=np.array(y_test).flatten()
def patient_searching():
    diagnosis=['good','ill']
    num=input("itner the Patient num u wanna predect\n from 1 to 112\n")
    num=int(num)
    num=num+1
    #144 is the number of rows in the test sample 
    if num in range(114):
        print('prediction is : ',diagnosis[y_pred[num]])
        print('ture value is :',diagnosis[y_test[num]])
    elif num > 112:
        print('you intered the wrong  ID')

print(patient_searching())



















