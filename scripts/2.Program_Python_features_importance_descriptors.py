# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 12:35:19 2023

@author: medisp-2
"""

#PROGRAM CALCULATING FEATURES IMPORTANCE TO BE USED IN MACHINE LEARNING

import pandas as pd
import numpy as np


#-------------------------------------------------------------------
def classyChoice(clChoice):
       
        if(clChoice==0):
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=10)

        elif(clChoice==1):
            from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
            clf = DecisionTreeClassifier()
        
        elif(clChoice==2):
            from xgboost import XGBClassifier
            clf = XGBClassifier()

        elif(clChoice==3):
           from sklearn.ensemble import ExtraTreesClassifier
           clf = ExtraTreesClassifier(n_estimators=10, max_depth=None,
                                      min_samples_split=2, random_state=0)
        elif(clChoice==4):
           from sklearn.ensemble import AdaBoostClassifier as c
           clf = c(n_estimators=100)

        elif(clChoice==5):
           from sklearn.ensemble import GradientBoostingClassifier as c
           clf = c(n_estimators=100,learning_rate=1.0,max_depth=1, random_state=0)


        return(clf)    
        

def split_classes(X,y):
    class1=X[y==0]
    class2=X[y==1]
    return(class1,class2)  



def read_data(fileName,labels,id_file):
    df=pd.read_csv(fileName)
    n = 7
    df.drop(columns=df.columns[:n], 
            axis=1, 
            inplace=True)

    #CLEAN THE DATA
    #drop columns with theshold of zeros
    thresh = 20
    to_drop = df.eq(0).rolling(thresh).sum().eq(thresh).any()
    df=df.loc[:, ~to_drop]

    df=df.dropna() 
    df=df.fillna(0)
    df1=df.iloc[:,1:]. apply(lambda x: (x - x. min()) / (x. max() - x. min()))
    df=pd.concat([df.iloc[:,0],df1],axis=1)

    fNames=df.columns;fNames=np.asarray(fNames)
    Label=fNames[0]
    print(df.iloc[0,0:7])
    print(fNames)
    # U.RETURN()
    X1=df[df[Label]=='Active']
    X2=df[df[Label]=='Not Active']
    X1=np.asarray(X1);X2=np.asarray(X2)
    X1=np.asarray(X1);X2=np.asarray(X2)
    fNames=df.columns;fNames=np.asarray(fNames)
    XX1=X1[:,1:];XX2=X2[:,1:];fNames=fNames[1:]
    ''' ------------------IMPORTANT FOR TESTS OF UNEVEN DATA TO WORK----------------'''
    XX1=np.asarray(XX1,dtype='float64');XX2=np.asarray(XX2,dtype='float64');fNames=np.asarray(fNames)
    ''' ------------------END OF IMPORTANT FOR TESTS OF UNEVEN DATA TO WORK----------------'''
    XX=np.concatenate([XX1,XX2]);N1=np.size(XX1,0);N2=np.size(XX2,0)
    yy1=np.zeros(N1);yy2=np.ones(N2);
    yy=np.concatenate([yy1,yy2])
    return(XX,yy,fNames)
    
#function to claculate the descriptor's importance    
def features_importance(X,y,fNames,clChoice):
    # print('')
    rng = np.random.RandomState(1951)
    model = classyChoice(clChoice)
    X=np.asarray(X,dtype='float64')
    model.fit(X, y)
    importance = model.feature_importances_    
    imp=np.sort(importance)[::-1]
    orig_indx=np.argsort(importance)[::-1]
    X1=X[:,orig_indx]
    [class1,class2]=split_classes(X1,y)
    fNames=fNames[orig_indx]
    return(class1,class2,fNames)   
    
#--------------------------------------------------------------------
#function to rank descriptors by frequency of appearance
def freq(my_str):
 #from https://www.geeksforgeeks.org/find-frequency-of-each-word-in-a-string-in-python/
    '''1 break the string into list of words'''
    str = my_str        
    str2 = []
    '''2. loop till string values present in list str'''
    for i in str:            
        '''3. # checking for the duplicacy'''
        if i not in str2:
            '''#5.3. insert value in str2'''
            str2.append(i)
             
    for i in range(0, len(str2)):
        freqs.append(str.count(str2[i]))
    return(str2,freqs)    

#--------------------------------------------------
#function to store the original id of the features with importance
def feat_impo_id(features_with_importance,fNames):
    id_fwi=[]
    for i in range (0,len(features_with_importance)):
        for j in range (0,len(fNames)):
            if(features_with_importance[i]==fNames[j]):
                  id_fwi.append(j)   
    return(id_fwi)
    
#-----------------------------------------------------------
#-------------------- MAIN PROGRAM -------------------------
#-----------------------------------------------------------
# import numpy as np
import moduleUtils as U
''' ---------------------READ DATA------------------------'''
U.cls()

'''1. Read the data file with descriptors (or features as names in the ML world'''

fileNames=['test_file_labeled_descriptors.csv']


my_file=fileNames[0]
data_files=[my_file]
id_file=0
path=''
fileName=data_files[id_file]
labels=['Active','non-Active']
X,y,fNames=read_data(path+fileName,labels,id_file)
print(np.shape(X))

'''2. Define the number of repetitions or epochs'''
number_of_epochs=10 #it may take some computer processing time for each epoch
number_of_epochs=int(input('give number of epochs:'))

X=np.asarray(X);y=np.asarray(y);fNames=np.asarray(fNames)

'''3.Define the 6 classifiers to be employed  '''
namesOfClassifiers = ['0:RandomForest_classifier', ' 1:CART_classifier','2:XGBoost_classifier',
                      '3:ExtraTreesClassifier','4:AdaBoost_classifier',
                      '5:GradientBoostClassfier' ] 
''' 4. Calculate the frequencies of appearance of the features with importance'''
feats_freq=[]
number_of_epochs=number_of_epochs
t1=U.tic()
''' repeat for a number of epochs '''
for epochs in range (0,number_of_epochs):
    print('**** epoch: ',epochs)    
    
    for classies in range (0,6):
        clChoice=classies
        '''4.1 calculate and sort by importance the features (descriptors) for each classifier'''
        (cl1,cl2,fNames_sorted)=features_importance(X,y,fNames,clChoice)
        k=fNames_sorted[0:20]
        feats_freq.append(k)

feats_freq=np.asarray(feats_freq)
feats_freq_1d=feats_freq.flatten

feats_freq_sorted=np.sort(feats_freq_1d())
import sys
np.set_printoptions(threshold=sys.maxsize)
'''#5. find unique features and their frequency'''
freqs=[]
names, freqs=freq(list(feats_freq_sorted))
df=pd.DataFrame([names,freqs])
df=df.T
df.columns=['features','frequency']
df1=df.sort_values(by='frequency',ascending=False)
z=df1['frequency'].values>number_of_epochs
FEATS=df1[z==True]
print(FEATS)
print(list(FEATS['features']))
print('number of features',len(FEATS['features']))

'''Save most important features (descriptors)'''
feats_of_high_importance=FEATS['features'].values
id_fwi=feat_impo_id(feats_of_high_importance,fNames)
id_fwi=np.asarray(id_fwi,int)
print('\nfeatures with high importance  found:',id_fwi)
print('\nnumber of high importance features:', np.shape(id_fwi))
fl=data_files[id_file][:-4]+'_features-importance_ids.txt'
fl2=data_files[id_file][:-4]+'_features-importance_Names.txt'
np.savetxt(fl,id_fwi,fmt='%d')
np.savetxt(fl2,feats_of_high_importance,fmt='%s')
print("descriptors' file has been created")

U.toc(t1,'')