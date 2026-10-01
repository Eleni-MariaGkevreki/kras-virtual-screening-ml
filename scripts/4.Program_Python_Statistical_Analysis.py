# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 15:57:29 2023

@author: medisp-2
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 14:29:32 2023

@author: medisp-2
"""
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import scipy
import moduleUtils as U
import random
random.seed(10)


#PROGRAM FOR THE STATISTICAL ANALYSIS OF FEATURES (DESCRIPTORS)
#---------------------------------------------------------------------------    
def read_data(fileName,labels,id_file):
    df=pd.read_csv(fileName)
    n = 7
    df.drop(columns=df.columns[:n], 
            axis=1, 
            inplace=True)
  
    #drop columns with thesh zeros
    thresh = 20
    to_drop = df.eq(0).rolling(thresh).sum().eq(thresh).any()
    df=df.loc[:, ~to_drop]
    df=df.dropna() 
    df=df.fillna(0)
    df1=df.iloc[:,1:]. apply(lambda x: (x - x. min()) / (x. max() - x. min()))
    df=pd.concat([df.iloc[:,0],df1],axis=1)
    fNames=df.columns;fNames=np.asarray(fNames)
    Label=fNames[0]
    X1=df[df[Label]=='Active']
    X2=df[df[Label]=='Not Active']
    print('\n-----------\n',np.shape(X1));print(np.shape(X2));
    X1=np.asarray(X1);X2=np.asarray(X2)
    fNames=df.columns;fNames=np.asarray(fNames)
    XX1=X1[:,1:];XX2=X2[:,1:];fNames=fNames[1:]

    ''' ------------------IMPORTANT FOR TESTS OF UNEVEN DATA TO WORK----------------'''
    XX1=np.asarray(XX1,dtype=float);XX2=np.asarray(XX2,dtype=float);fNames=np.asarray(fNames)
    ''' ------------------END OF IMPORTANT FOR TESTS OF UNEVEN DATA TO WORK----------------'''
    XX=np.concatenate([XX1,XX2]);N1=np.size(XX1,0);N2=np.size(XX2,0)
    yy1=np.zeros(N1);yy2=np.ones(N2);
    print(np.shape(yy1));print(np.shape(yy2))
    yy=np.concatenate([yy1,yy2])
    print(np.shape(fNames))
    return(XX,yy,fNames)

#############################################################################
'''                       MAIN PROGRAM                                    '''
#############################################################################
import scipy.stats as stats
import pandas as pd
U.cls()

''' 1. Read Data Files'''
''' 1a. Read the csv file of the data with descriptors'''

data_files=['test_file_labeled_descriptors.csv']

id_file=0
fileName=data_files[id_file]
st_type=['Active','Not Active'];labels=st_type
X,y,fNames=read_data(fileName,labels,id_file)
print(np.shape(X),np.shape(y),np.shape(fNames))


'''1b. Read best features file created in the feature importance program (2nd program)'''

fl2=data_files[id_file][:-4]+'_features-importance_Names.txt'
# test_file_labeled_descriptors_features-importance_ids.txt
HE_fNames=np.loadtxt(fl2,'str')
HE_features=np.asarray(HE_fNames)



''' 2.STATISTICAL ANALYSIS'''
'''2.a. function for plotting box plots'''
def BXPLT(x, Labels, Title):
    plt.boxplot(x, labels=Labels,showfliers=False,whis=1)
    plt.title(Title)
    plt.grid()
    plt.show()

''' 2b. For each feature test its Statistical Significance w.r.t. " Active - Not Active " classes '''
from scipy import stats
icfeats=0;fN=[];pval=[]
for j in range(0,len(fNames)):
    if(fNames[j] in HE_fNames):
        x=X[:,j]
        (Fstatistic1, p) = sc.stats.mannwhitneyu(x[y==0], x[y==1])
        print(p)
        if(p<0.05):
            Labels=['Active', 'Not Active'];Title='Feature with SSD: '+ fNames[j] + ', p= '+str('%4.3e'%p)
            print(np.shape(X))
            fN.append(fNames[j])
            pval.append('%4.3e'%p)
            icfeats=icfeats+1
            BXPLT([x[y==0], x[y==1]], Labels, Title)

''' 2b. print out the Statistical Analysis Findings '''
print('Number of features with SSD:',icfeats)

DF_fN=pd.DataFrame([fN,pval]);DF_fN=DF_fN.T
DF_fN.columns=['Feature_Names','p<0.05']
print('**** Note: SSD stands for Statistically Significant Difference ****')
print(DF_fN)




