#PROGRAM FOR MACHINE LEARNING -SYSTEM-DEPLOYMENT 
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
#function to read the the csv, to clean them from null spaces or data with many consecutive 0s,
#put the actual data values in XX for both classes, form column array yy (vector) for labeling 
# each data line (corresponding to one Molecule), and forming the fNames vector with the names 
# of the descriptors    
def read_data(fileName):
    df=pd.read_csv(fileName)
    
    moleculeNames=df['Molecule ChEMBL ID']
    moleculeNames=np.asarray(moleculeNames,str)
    '''1.delete the first 6 columns containing the data verbal info '''
    n = 7
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
    df=pd.concat([df.iloc[:,0],df1],axis=1)

    
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
    return(XX,yy,fNames,moleculeNames)


# def read_data(fileName,labels,id_file):
#     df=pd.read_csv(fileName)
#     # df.to_excel('ttt.xlsx')
#     # U.RETURN()
#     '''1.delete the first 6 columns containing the data verbal info '''
#     n = 7
    
    
#     # print(stValcol)
    
    
#     # U.RETURN()
#     df.drop(columns=df.columns[:n], 
#             axis=1, 
#             inplace=True)
#     '''2 drop columns below a theshold of  zeros'''
#     thresh = 20
#     to_drop = df.eq(0).rolling(thresh).sum().eq(thresh).any()
#     df=df.loc[:, ~to_drop]
#     '''3 drop columns with not avalailable data'''
#     df=df.dropna() 
#     '''#4 fill with zero not existing data values '''
#     df=df.fillna(0)

#     df1=df.iloc[:,1:]. apply(lambda x: (x - x. min()) / (x. max() - x. min()))
#     '''5. Form the X,y,fNames data used by the ML algorithms'''    
#     df1=df.iloc[:,1:]
#     df=pd.concat([df.iloc[:,0],df1],axis=1)
#     fNames=df.columns;fNames=np.asarray(fNames)
#     Label=fNames[0]
#     X1=df[df[Label]=='Active']
#     X2=df[df[Label]=='Not Active']
    
#     X1=np.asarray(X1);X2=np.asarray(X2)
#     fNames=df.columns;fNames=np.asarray(fNames)
#     XX1=X1[:,1:];XX2=X2[:,1:];fNames=fNames[1:]
#     ''' ------------------IMPORTANT FOR TESTS OF UNEVEN DATA TO WORK----------------'''
#     XX1=np.asarray(XX1,dtype=float);XX2=np.asarray(XX2,dtype=float);fNames=np.asarray(fNames)
#     ''' ------------------END OF IMPORTANT FOR TESTS OF UNEVEN DATA TO WORK----------------'''
#     XX=np.concatenate([XX1,XX2]);N1=np.size(XX1,0);N2=np.size(XX2,0)
#     yy1=np.zeros(N1);yy2=np.ones(N2);
#     yy=np.concatenate([yy1,yy2])
#     return(XX,yy,fNames)

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

def form_external_evaluation_dataset (X,y):  
    Xf=pd.DataFrame(X);yf=pd.Series(y)
    # print(Xf.shape)
    dim1=np.size(X,0);print(dim1)
    # U.RETURN()
    prc=40/dim1 #form ratio corresponding to 40 molecules as test data 
    XTRAIN, XTEST, yTRAIN, yTEST = train_test_split(Xf,yf, test_size=prc,stratify=y,random_state=1234)#
    XTESTfeat_ids=XTEST.index.to_list()
    XTEST_file='XTEST_saved_validation_data.csv'
    yTEST_file='yTEST_saved_validation_data.csv'
    np.savetxt(XTEST_file,XTEST)
    np.savetxt(yTEST_file,yTEST)
    return (XTRAIN, yTRAIN,XTEST_file,yTEST_file,XTESTfeat_ids)

#---------------------------------------------------------------------- 
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
#================================================================
''''# 1.7.  Data normalization function: f'= (f-mean)/std'''
def normalizeData(X):
    nPatts=np.size(X,0);nFeats=np.size(X,1)
    M=np.zeros(nFeats);S=np.zeros(nFeats)
    for i in range (nFeats):
        M[i]=np.mean(X[:,i])
        S[i]=np.std(X[:,i])

    for i in range(nPatts):
        for j in range(nFeats):
            X[i,j]=(X[i,j]-M[j])/S[j]
   
    ddf=pd.DataFrame(X)
    ddf=ddf.fillna(0)
    X=ddf.values;X=np.asarray(X)        
    return(X,M,S)       
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

ID_FILE=0

my_file=fileNames[ID_FILE]
data_files=[my_file]
# data_files=['test_file_with_descriptors_2.csv']
id_file=0
path=''
fileName=data_files[id_file]
print(fileName)

'''# 124. Read the csv datafile '''
labels=['Active','non-Active']
X,y,fNames,moleculeNames=read_data(path+fileName)
'''#1.2. Choose classifier '''
namesOfClassifiers = ['0:RandomForest_classifier', ' 1:CART_classifier','2:XGBoost_classifier',
                      '3:ExtraTreesClassifier','4:AdaBoost_classifier',
                      '5:GradientBoostClassfier' ] 

print('Available classifiers: ')
print(list(namesOfClassifiers))

clChoice=0
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

choice_of_smote=1
labels=['Active','Not Active'];

'''Split the data into  THE DESIGN dataset and the External VALIDATION DATASET'''

print(X.shape)
XDESIGN,yDESIGN,XTEST_file,yTEST_file,XTESTfeat_ids=form_external_evaluation_dataset (X,y)



X=np.copy(XDESIGN);y=np.copy(yDESIGN)

'''#1.4.Read features with importance stored by the 2nd program'''
fl=data_files[id_file][:-4]+'_features-importance_ids.txt'
id_fwi=np.loadtxt(fl)
id_fwi=np.asarray(id_fwi,int)

fl2=data_files[id_file][:-4]+'_features-importance_Names.txt'
fNames=np.loadtxt(fl2,'str')
X=X[:,id_fwi]  


Features2use=len(id_fwi)
'''1.5. Keep a copy of original data''' 
Xst=np.copy(X);yst=np.copy(y);fNamesst=np.copy(fNames)
#========================================================================

X=Xst;y=yst;fNames=fNamesst;results=[]
'''#1.8. define the size of plots '''
# plt.figure(figsize=(12,10))
'''---------------------------------------------------------------'''
'''-----------------     EPOCHS    -------------------------------'''
'''---------------------------------------------------------------'''
'''2. START THE MACHINE LEARNING PROCESS '''

misclas=[];ACR=[];
plt.figure(figsize=(12,10))
# U.RETURN()
results=[]
for epoch in range(0,1):
      X=Xst;y=yst;fNames=fNamesst;
      sz=200;np.size(X,0)
      if(np.size(X,0)<sz):
          X,y=data_augmentation(X,y,sz)
      print('\n-----------------------------------------------  ')
      print('epoch: ', epoch)  

      '''#2.1. Class equalization '''
      if(choice_of_smote!=1):
          print('NO CLASS EQUALIZATION by the SMOTE METHOD ')
          X_e=np.copy(X);y_e=np.copy(y)
      else:    
            from imblearn.over_sampling import SMOTE
            print('SMOTE METHOD of equalizing classes')
            oversample = SMOTE()
            X_e, y_e = oversample.fit_resample(X, y)
      X_e = np.asarray(X_e);y_e = np.asarray(y_e);
              
      
      '''2.2. Define the starting parameters        '''
      model = classyChoice(clChoice)
      t1 = U.tic()
      bestFeatures = []
      maxAccuracy = 0;
      bestTT = np.zeros((2, 2), int)  # for 2 classes
      
      '''2.3. DATA SPLIT FOR MODEL EVALUATION BY EXTERNAL DATA SET '''
      #************************EPOCH RANDOM DATA SPLIT *********************
      X_TRAIN, X_TEST, y_TRAIN, y_TEST = train_test_split(X_e, y_e, test_size=0.30)
              
      ''' Normalize features of X_TRAIN and from the M and S normalize X_TEST see function in 1.7 step'''
      X_TRAIN,M,S=normalizeData(X_TRAIN)
      ''' use the mean and standard deviation of the X_TRAIN dataset to normalize the X_TEST dataset'''
      X_TEST=(X_TEST-M)/S
      print(np.shape(X_TEST),np.shape(M),np.shape(S))

 
      ''' 2.4. for an increasing number of features (up to the chosen maximum) '''
      for n_feats in range (1,n_features+1): 
          feats=np.arange(0,n_feats,1)
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
      fl2=data_files[id_file][:-4]+'_best features for descrimination_Names.txt'
      np.savetxt(fl2,fNames[bestFeatures],fmt='%s')
      fl1=data_files[id_file][:-4]+'_best features for descrimination_ids.txt'
      np.savetxt(fl1,bestFeatures)
    
      fl_Means=data_files[id_file][:-4]+'_Descriptor-Means.txt'
      np.savetxt(fl_Means,M)
    
      fl_STDs=data_files[id_file][:-4]+'_Descriptor-STDs.txt'
      np.savetxt(fl_STDs,S)
      
        
      U.toc(t1,'epoch time')
      print('')
            
      # results.append([np.round(ACCURACY_TT,3), np.round(SPEC,3), np.round(SENS,3), TT[0,0],TT[0,1],TT[1,0],TT[1,1], np.round(ROC_AUC,3), len(bestFeatures)])
      results.append([np.round(ACCURACY_TT,3), np.round(SPEC,3), np.round(SENS,3),np.round(ROC_AUC,3), len(bestFeatures)])
    
      classif_names = ['RandomForest_classifier', 'CART_classifier','XGBoost_classifier',
                       'ExtraTreesClassifier','AdaBoost_classifier','GradientBoostClassier' ] 
      namesOfClassifiers = ['0:RandomForest_classifier', ' 1:CART_classifier','2:XGBoost_classifier',
                            '3:ExtraTreesClassifier','4:AdaBoost_classifier','5:GradientBoostClassier' ] 
    
      DF_results=pd.DataFrame(results)
      DF_results.columns=['Total acc.', '~Active acc.', 'Active acc.','ROC_AUC', 'No of features']
      
      ''' 2. End of current epoch'''
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
print('\n\n\n ================descriptors-impact evaluation====================')
plot_id=1#
'''if  plot_id=1 then plot bar-plots of the features impact to classification'''
feat_imp_=used_features_importance(X[:,HE],y,clChoice,fNames,feats,plot_id ) 
# print('features-importance for classification: ',feat_imp_)
# print(fNames)
# plt.show()
  
'''#4.Save the designed model for use  '''
'''4.1 Save designed model on disk '''
import pickle
filename =data_files[id_file][:-4]+'_best model.sav'
print('\nModel file name: ',filename)
Model=MODEL
'''4.2 first evaluate the accuracy using the TEST dataset'''
Results=Model.score(X_TEST[:,bestFeatures],y_TEST)
print('Use of Internally Designed  Model Results',Results)
'''4.3 Save the model'''
pickle.dump(MODEL, open(filename, 'wb'))
  
''' 4.4. Read the model from the disk'''
Mdl = pickle.load(open(filename, 'rb'))
'''4.5. Calculate the accuracy of read Model on the TEST dataset'''
Results=Mdl.score(X_TEST[:,bestFeatures],y_TEST)
print('Loaded Model Results: ', Results)
  
  
  
'''4.6. Calculate the accuracy of the read Model (Mdl.) on the EXTERNAL TEST dataset'''
X_EXT=np.loadtxt(XTEST_file)
y_EXT=np.loadtxt(yTEST_file)
X_EXT=X_EXT[:,id_fwi]
M_EXT=np.loadtxt(fl_Means)
S_EXT=np.loadtxt(fl_STDs)
'''#Normalize the external data set'''
X_EXT=(X_EXT-M_EXT)/S_EXT
''' Classify the external data'''
EXT_Results=Mdl.score(X_EXT[:,bestFeatures],y_EXT)
print('Loaded Model USING EXTERNAL UNSEEN DATA: Results: ', EXT_Results)
  
PREDICTED_y = Mdl.predict(X_EXT[:,bestFeatures]) #CLASS PREDICTION  
EXPECTED_y=y_EXT
dif=np.abs(EXPECTED_y-PREDICTED_y);
EXT_y_accuracy=MODEL.score(X_EXT[:,bestFeatures],y_EXT)
PREDICTED_y=np.asarray(PREDICTED_y,int);EXPECTED_y=np.asarray(EXPECTED_y,int);
dif=np.abs(PREDICTED_y-EXPECTED_y);dif=np.asarray(dif,int)
  
DDF=pd.DataFrame([moleculeNames[XTESTfeat_ids],
                  PREDICTED_y,
                  EXPECTED_y,
                  dif])
DDF=DDF.T
DDF.columns=['Molecule','predicted_class', 'expected_class','misclassified(=1)']
print(DDF.to_string())
print('\n------------ MISCLASSIFIED MOLECULES---------:\n')
print(DDF[DDF['misclassified(=1)']==1])

print('number of misclassified %4d out of %d molecules in Total, accuracy %.2f '% 
 (DDF[DDF['misclassified(=1)']==1].shape[0],len(DDF['misclassified(=1)']==0),EXT_Results))
ACR.append(EXT_Results)
arr=DDF.values
arr=np.asarray(arr)
ar1=np.asarray(DDF.index[arr[:,3]==1],int)
ar2=np.asarray(DDF.values[ar1,0],str)
      

print('\n***mean accuracy of unknowndata classification: ', np.round(np.mean(ACR),2));print('')

#read sdf data files turned to csv with descriptors and classify them Active Not Active
#1.read the data
fileNames2=['test_file_hits_descriptors.csv']
path1=''
my_file=fileNames2[ID_FILE]
File=path1+my_file
print('\n processing file: %s ' % File)
print(' please wait.....')
data = pd.read_csv(File)
data=data.dropna() 
data=data.fillna(0)
fN=fNames[bestFeatures]
NamesSDF=data.columns;NamesSDF=np.asarray(NamesSDF)
# U.RETURN()
indxSDF=[]
for i in range(0,len(NamesSDF)):
    if(NamesSDF[i] in fN):
        indxSDF.append(i)

Mols=data.iloc[:,1];Mols=np.asarray(Mols)
X=data.values;X=np.asarray(X);X=X[:,3:];rmsd=X[:,2];
NamesSDF=NamesSDF[3:]        
ddf=pd.DataFrame(X)
ddf=ddf.dropna() 
ddf=ddf.fillna(0)
X=ddf.values;X=np.asarray(X)
XTEST=X[:,indxSDF]
XTEST,_,_=normalizeData(XTEST)
PREDICTED_y = Mdl.predict(XTEST)
dfSDF=pd.DataFrame([Mols,PREDICTED_y,rmsd])
dfSDF=dfSDF.T
dfSDF.columns=['Molecule', 'PREDICTED','rmsd'  ]
# print(dfSDF)
print('sum pred:',np.sum(PREDICTED_y))
dfSDF=dfSDF.sort_values(by=['PREDICTED','rmsd'])
print('\n',dfSDF[dfSDF['PREDICTED']==0.0].iloc[0:20,:])
print('\n********Predicted=0.0 is Active*********')
U.pause()
print('\n',dfSDF[dfSDF['PREDICTED']==1.0].iloc[0:20,:])
print('\n********Predicted=1.0 is Not Active*********')
end_sound(1000)

# Υπολογισμός του πλήθους για κάθε κλάση
counts = dfSDF['PREDICTED'].value_counts()

# Εμφάνιση των αποτελεσμάτων
print("\n=== Στατιστικά Στοιχεία Κατηγοριοποίησης ===")
print(f"Δραστικές ενώσεις (Active - 0): {counts.get(0, 0)}")
print(f"Μη δραστικές ενώσεις (Not Active - 1): {counts.get(1, 0)}")
print(f"Συνολικό πλήθος ενώσεων: {len(DDF)}")


# --- ΠΡΟΣΘΗΚΗ ΣΤΟ ΤΕΛΟΣ ΤΟΥ ΚΩΔΙΚΑ ---

# Φιλτράρισμα μόνο των δραστικών ενώσεων (PREDICTED == 0.0)
active_ligands_df = dfSDF[dfSDF['PREDICTED'] == 0.0]

# Αποθήκευση σε αρχείο Excel
output_filename = "active ligands.xlsx"
active_ligands_df.to_excel(output_filename, index=False)

print(f'\nΤο αρχείο "{output_filename}" δημιουργήθηκε με επιτυχία και περιέχει {len(active_ligands_df)} δραστικές ενώσεις.')