# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 15:34:07 2025

@author: medisp-2
"""

#PROGRAM FOR MACHINE LEARNING ANALYSIS
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
import sys

import moduleUtils as U
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score

U.cls()
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" 

#Function to plot the ROC curves
def calcROC(X_train, X_test, y_train, y_test,clChoice,Title,Acc):
#    https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
    
    if(clChoice==0):
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=10)
        clf.fit(X_train,y_train)
        y_score=clf.predict_proba(X_test)[:,1]

    elif(clChoice==1):
        from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
        clf = DecisionTreeClassifier(min_samples_split=30)
        clf.fit(X_train,y_train)
        y_score=clf.predict_proba(X_test)[:,1]
        
    elif(clChoice==2):
        from xgboost import XGBClassifier
        clf = XGBClassifier()
        clf.fit(X_train,y_train)
        y_score=clf.predict_proba(X_test)[:,1]
        
    elif(clChoice==3):
       from sklearn.ensemble import ExtraTreesClassifier
       clf = ExtraTreesClassifier(n_estimators=10, max_depth=None,
                                  min_samples_split=2, random_state=0)
       clf.fit(X_train,y_train)
       y_score=clf.predict_proba(X_test)[:,1]
    elif(clChoice==4):
       from sklearn.ensemble import AdaBoostClassifier as c
       clf = c(n_estimators=100)
       clf.fit(X_train,y_train)
       y_score=clf.predict_proba(X_test)[:,1]

    elif(clChoice==5):
       from sklearn.ensemble import GradientBoostingClassifier as c
       clf = c(n_estimators=100,learning_rate=1.0,max_depth=1, random_state=0)
       clf.fit(X_train,y_train)
       y_score=clf.predict_proba(X_test)[:,1]
        
    # Compute ROC curve and ROC area for each class
    from sklearn.metrics import roc_curve, auc
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    fpr,tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    
    # plt.figure(figsize=(8,6))
    lw = 2
    plt.plot(fpr, tpr, lw=lw, label='ROC curve (area = %0.6f), accuracy = %4.3f, descriptors : %2d' % (roc_auc,Acc, np.size(X_train,1) ) )
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    plt.title(Title)
    plt.legend(loc="lower right");plt.grid()
 
    return(roc_auc)    

#function to choose 1 classifier out of 6 classifiers 
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

#-------------------------------------------------------------------
# function to anounce the end of the ML design
def end_sound(dur):
    import winsound
    duration = dur  # milliseconds
    freq = 440  # Hz
    winsound.Beep(freq, duration)
#-------------------------------------------------------------------
def data_augmentation(X,y,sz):
    from sklearn.utils import resample
    from numpy.random import seed
    from numpy.random import normal
    
    y_0=np.zeros(len(y[y==0]))
    y_1=np.ones(len(y[y==1]))
    y=np.concatenate([y_0,y_1])
    seed(10)
    # X,y=resample(X,y, n_samples=int(len(y))*3, replace=True, stratify=y,random_state=1234)
    X,y=resample(X,y, n_samples=sz, replace=True, stratify=y,random_state=1234)
    X=np.asarray(X);y=np.asarray(y)
    # print(X[0:5,0:5])
    # print(np.shape(X))
    # U.RETURN()
    for II in range(np.size(X,1)):
        # for JJ in range(np.size(X,1)):
        gauss = normal(loc=0.0, scale=0.2, size=np.size(X,0))
        X[:,II]=X[:,II]+gauss
        # U.pause()
    # print('***********:',np.shape(X),np.shape(y))
    # U.pause()
    return X,y



#function to read the the csv, to clean them from null spaces or data with many consecutive 0s,
#put the actual data values in XX for both classes, form column array yy (vector) for labeling 
# each data line (corresponding to one Molecule), and forming the fNames vector with the names 
# of the descriptors    
def read_data(fileName,labels,id_file):
    df=pd.read_csv(fileName)
    # df.to_excel('ttt.xlsx')
    # U.RETURN()
    '''1.delete the first 6 columns containing the data verbal info '''
    n = 7
    
    
    # print(stValcol)
    
    
    # U.RETURN()
    df.drop(columns=df.columns[:n], 
            axis=1, 
            inplace=True)
    '''2 drop columns below a theshold of  zeros'''
    thresh = 20
    to_drop = df.eq(0).rolling(thresh).sum().eq(thresh).any()
    df=df.loc[:, ~to_drop]
    '''3 drop columns with not avalailable data'''
    df=df.dropna() 
    '''#4 fill with zero not existing data values '''
    df=df.fillna(0)

    df1=df.iloc[:,1:]. apply(lambda x: (x - x. min()) / (x. max() - x. min()))
    '''5. Form the X,y,fNames data used by the ML algorithms'''    
    df1=df.iloc[:,1:]
    df=pd.concat([df.iloc[:,0],df1],axis=1)
    fNames=df.columns;fNames=np.asarray(fNames)
    Label=fNames[0]
    X1=df[df[Label]=='Active']
    X2=df[df[Label]=='Not Active']
    
    X1=np.asarray(X1);X2=np.asarray(X2)
    fNames=df.columns;fNames=np.asarray(fNames)
    XX1=X1[:,1:];XX2=X2[:,1:];fNames=fNames[1:]
    ''' ------------------IMPORTANT FOR TESTS OF UNEVEN DATA TO WORK----------------'''
    XX1=np.asarray(XX1,dtype=float);XX2=np.asarray(XX2,dtype=float);fNames=np.asarray(fNames)
    ''' ------------------END OF IMPORTANT FOR TESTS OF UNEVEN DATA TO WORK----------------'''
    XX=np.concatenate([XX1,XX2]);N1=np.size(XX1,0);N2=np.size(XX2,0)
    yy1=np.zeros(N1);yy2=np.ones(N2);
    yy=np.concatenate([yy1,yy2])
    return(XX,yy,fNames)
#---------------------------------------------------------------  
#function to calculate, rank in descending order, and plot a bar-diagram  the importance 
#or impact of the descriptors employed in the best ML-system design 
def used_features_importance(x,y,clChoice,fNames,feats,plot_id):
    model=classyChoice(clChoice)
    model.fit(x, y)
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    indices =np.asarray(indices)
    # Rearrange feature names so they match the sorted feature importances
    fN=fNames[feats]
    names = [fN[i] for i in indices]
    feat_imp=np.round(importances[indices],2)
    feat_imp=feat_imp[feat_imp>0.02]
    KK=len(feat_imp)
    
    feat_imp=np.asarray(feat_imp)
    if(plot_id==1):
        plt.figure(figsize=(15, 8))
        plt.title("Descriptors-Importance (impact on classification) ",size=16,color='m')
        
        plt.bar(range(x[:,0:KK].shape[1]), feat_imp,width=0.8,color='lightgreen',edgecolor='red' )
        for i in range(len(feat_imp)):
          plt.text(i,feat_imp[i],feat_imp[i])
        fntsz=12
        plt.xticks(range(x[:,0:KK].shape[1]), names[0:KK], rotation=60,fontsize=fntsz)
        plt.xlabel("Descriptors",color='b',size=14)
        plt.ylabel("Descriptor-Importance",color='b',size=14)
        plt.grid()
    return(feat_imp)
#----------------------------------------------------------------------        
'''
    
#-----------------------------------------------------------
#-------------------- MAIN PROGRAM -------------------------
#-----------------------------------------------------------
'''
import moduleUtils as U
''' ---------------------READ DATA------------------------'''
U.cls()

'''#1. READ AND PREPARE THE DATA FOR MACHINE LEARNING'''


''' 1.1. Define the csv file of the data with descriptors'''
fileNames=['test_file_labeled_descriptors.csv']
fileName=fileNames[0]
data_files=[fileName]
id_file=0
path=''
fileName=data_files[id_file]
labels=['Active','non-Active']
X,y,fNames=read_data(path+fileName,labels,id_file)
print(np.shape(X))
# U.RETURN()




'''#1.2. Choose classifier '''
namesOfClassifiers = ['0:RandomForest_classifier', ' 1:CART_classifier','2:XGBoost_classifier',
                      '3:ExtraTreesClassifier','4:AdaBoost_classifier',
                      '5:GradientBoostClassfier' ] 


print('Available classifiers: ')
print(list(namesOfClassifiers))

clChoice=1
clChoice=int(input('give id number of classifier (0 to 5): '))
if(clChoice not in (0,1,2,3,4,5)):
    print('Classifier not available')
    U.RETURN()

'''#.1.3. Choose number of features'''
n_features=10
n_features=int(input('\ngive maximum number of features to be used>2** (e.g. 5,10,15,20) : '))
if(n_features<2):
   n_features=3

print(n_features)


# choice_of_smote=int(input('\n for class equalization:1  otherwise= any number: '))
choice_of_smote=1
# fileName=data_files[id_file]
labels=['Active','Not Active'];

# '''# 1.4. Read the csv datafile '''
# X,y,fNames=read_data(fileName,labels,id_file)

sz=len(y)
if(np.size(X,0)<sz):
    X,y=data_augmentation(X,y,sz)
elif(np.size(X,0)>1000):
    print('downsize')    
print(X.shape)
# U.RETURN()    
'''#1.5.Read features with importance stored by the 2nd program'''
fl=data_files[id_file][:-4]+'_features-importance_ids.txt'

id_fwi=np.loadtxt(fl)
id_fwi=np.asarray(id_fwi,int)

fl2=data_files[id_file][:-4]+'_features-importance_Names.txt'
fNames=np.loadtxt(fl2,'str')
X=X[:,id_fwi]  


Features2use=len(id_fwi)
'''1.6. Keep a copy of original data''' 
Xst=np.copy(X);yst=np.copy(y);fNamesst=np.copy(fNames)


'''1.7. #Normalize data'''
from sklearn import preprocessing
X=preprocessing.normalize(X,axis=0)

X=Xst;y=yst;fNames=fNamesst;results=[]
'''#1.8. define the size of plots '''
plt.figure(figsize=(12,10))


'''---------------------------------------------------------------'''
'''-----------------     EPOCHS    -------------------------------'''
'''---------------------------------------------------------------'''
'''2. START THE MACHINE LEARNING PROCESS '''
for epoch in range(0,5):
      print('\n-----------------------------------------------  ')
      print('epoch: ', epoch)  
      sz=200;np.size(X,0)
      if(np.size(X,0)<sz):
          X,y=data_augmentation(X,y,sz)
      print(np.shape(X))    
      '''#2.1. Class equalization '''
      if(choice_of_smote==0):
        print('NO CLASS EQUALIZATION by the SMOTE METHOD ')
        X_e=np.copy(X);y_e=np.copy(y)
      else:    
          from imblearn.over_sampling import SMOTE
          print('SMOTE METHOD of equalizing classes')
          oversample = SMOTE()
          X_e, y_e = oversample.fit_resample(X, y)
      X_e = np.asarray(X_e);y_e = np.asarray(y_e);
      
      # print(np.shape(X_e[y_e==0]),np.shape(X_e[y_e==1]))
      # U.RETURN()            
      
      '''2.2. Define the starting parameters        '''
      model = classyChoice(clChoice)
      t1 = U.tic()
      bestFeatures = []
      maxAccuracy = 0;
      bestTT = np.zeros((2, 2), int)  # for 2 classes
      
      '''2.3. DATA SPLIT FOR MODEL EVALUATION BY EXTERNAL DATA SET '''
      #************************EPOCH RANDOM DATA SPLIT *********************
      X_TRAIN, X_TEST, y_TRAIN, y_TEST = train_test_split(X_e, y_e, test_size=0.30)
      
      ''' 2.4. for an increasing number of features (up to the chosen maximum) '''
      for n_feats in range (1,n_features+1): 
          feats=np.arange(0,n_feats,1)
          print('\r'+str(feats),end='')
          HE=feats #HE is used later for descriptor impact 
          feats = np.asarray(feats)
          X_t_e = X_TRAIN[:,feats]
          y_t_e = y_TRAIN
          
          '''2.5. Internal evaluation by Repeated K_Fold of the model's accuracy  
          using the training dataset '''  
          from numpy import mean
          cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)
          scores = cross_val_score(model, X_t_e, y_t_e, scoring='accuracy', cv=cv, n_jobs=-1)
          accuracy=mean(scores)
          
          
          '''2.6. Temporary keep the highest of the so far achieved accuracies and associated parameters'''
          if(accuracy>maxAccuracy):
              maxAccuracy=accuracy;
              bestFeatures = feats
              MODEL=model
              plot_id=0
              feat_imp=used_features_importance(X_t_e,y_t_e,clChoice,fNames,feats,plot_id) 
      
      '''2.5. EVALUATE MODEL ACCURACY USING THE EXTERNAL DATASET (X_TEST)'''   
      print('\n-------Epoch %d model evaluation using the TEST dataset:'% epoch)
      print('best features id combination: ', bestFeatures)    
      print('best features Names combination: ', fNames[bestFeatures])
      EXPECTED_y  = y_TEST
      MODEL.fit(X_TRAIN[:,bestFeatures],y_TRAIN)
      PREDICTED_y = MODEL.predict(X_TEST[:,bestFeatures]) #CLASS PREDICTION  
      from sklearn.metrics import confusion_matrix #usage for calculating accuracies axplicitely
      TT=confusion_matrix(EXPECTED_y, PREDICTED_y)
      ACCURACY_TT=(TT[0,0]+TT[1,1])/np.sum(TT)
      SPEC=(TT[0,0])/(TT[0,0]+TT[0,1])# Active
      SENS=(TT[1,1])/(TT[1,1]+TT[1,0])#Not Active
      print('\naccuracy= ',np.round(ACCURACY_TT,3), '\nfeat_names=' ,fNames[bestFeatures])
      print('\nTT: \n', TT)
      print('\nTT accuracy: ', np.round(ACCURACY_TT,3),end='')
      print(', ~Active: ', np.round(SPEC,3),', Active: ',np.round(SENS,3))
      
      Title=fileName[:-4]+'\nClassifier used: %s ' % namesOfClassifiers[clChoice][2:]  
      ''' PLOT ROC-curve demonstrating the accuracy of classification of the external TEST DATA SET'''
      ROC_AUC= calcROC(X_TRAIN[:,bestFeatures], X_TEST[:,bestFeatures], y_TRAIN, y_TEST,clChoice,Title,ACCURACY_TT)
      plt.grid()
      ROC_val=np.round(ROC_AUC,5)
      print('ROC_AUC= ',np.round(ROC_val,3))
      print('Classifier employed: ',namesOfClassifiers[clChoice])
      
      U.toc(t1,'epoch time')
      
      results.append([np.round(ACCURACY_TT,3), np.round(SPEC,3), np.round(SENS,3), TT[0,0],TT[0,1],TT[1,0],TT[1,1], np.round(ROC_AUC,3)])
    
      classif_names = ['RandomForest_classifier', 'CART_classifier','XGBoost_classifier',
                       'ExtraTreesClassifier','AdaBoost_classifier','GradientBoostClassier' ] 
      namesOfClassifiers = ['0:RandomForest_classifier', ' 1:CART_classifier','2:XGBoost_classifier',
                            '3:ExtraTreesClassifier','4:AdaBoost_classifier','5:GradientBoostClassier' ] 
    
      DF_results=pd.DataFrame(results)
      DF_results.columns=['Total acc.', '~Active acc.', 'Active acc.', '~Active true','~Active missed','Active missed','Active true','ROC_AUC']
      
      ''' 2.6. End of current epoch'''
''' ************  END OF ALL EPOCHS *****************'''
'''
3.Print out the Final Results
#**********************************************************************
#***************************PRINT RESULTS AND FIGURES  *****************
#**********************************************************************
'''
DF_text2excel=classif_names[clChoice]+'_5fold_results' +'_.' + 'xlsx'
print(DF_text2excel)
print(DF_results.to_string())

plt.grid()
fig_file='ROC_curves_'+classif_names[clChoice]+'_'+'.png'
print(fig_file)
print(' mean_Accuracy: %4.3f' % DF_results['Total acc.'].mean() , end='')
print(', mean_~Active accuracy: %4.3f' % DF_results['~Active acc.'].mean(), end='')
print(', mean_Active accuracy: %4.3f' % DF_results['Active acc.'].mean(),end='')
print(', mean_ROC_auc: %4.3f' % DF_results['ROC_AUC'].mean())


print('best features:', fNames[bestFeatures])
end_sound(1000)
print('\n\n\n ================descriptors-impact evaluation====================')
plot_id=1#
'''if  plot_id=1 then plot bar-plots of the features impact to classification'''
feat_imp_=used_features_importance(X[:,HE],y,clChoice,fNames,feats,plot_id ) 
