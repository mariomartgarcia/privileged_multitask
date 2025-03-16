#from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
import warnings
from sklearn.datasets import  load_breast_cancer
from sklearn.impute import KNNImputer
from scipy.io import arff
import re
#from river.datasets import Phishing
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
# Disable all warnings
warnings.filterwarnings("ignore")



#--------------------------------------------------------
#1. PHISHING | (1250x9) | Binario

def phishing(from_csv = False):
    if from_csv:
        df = pd.read_csv('data/phishing.csv', index_col = False)
        y = df['0']
        X = df.drop(['Unnamed: 0', '0'], axis = 1)
        return X,y
    else:
        dataset = Phishing()
        X = pd.DataFrame()
        y = []
        for xx, yy in dataset.take(5000):
            X = pd.concat([X, pd.DataFrame([xx])], ignore_index=True)
            y.append(yy)
        y = pd.Series(y)*1
        return X, y



#--------------------------------------------------------
#2. DIABETES | (768x8) | Binario

def diabetes():
    df = pd.read_csv('data/diabetes.csv')
    X = df.drop('Outcome', axis = 1)
    y = df['Outcome']
    return X, y


#--------------------------------------------------------
#3. BREAST CANCER | (569x30) | Binario

def breast_cancer():
    bc = load_breast_cancer()
    df_bc = pd.DataFrame(data = bc.data, columns = bc.feature_names)
    
    df_bc['output'] = bc.target

    X = df_bc.drop('output', axis = 1)
    y = df_bc['output']
    
    return X, y
'''
#--------------------------------------------------------
#4. DRY BEAN | (13611x16) | 7 Classes


def dry_bean():
    database = fetch_ucirepo(id=602) 
    X = database.data.features 
    y = database.data.targets['Class']
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    return X, y

#--------------------------------------------------------
#5. WINE  | (6497x11)  | Binario, puede ser multiclase

def wine():
    database = fetch_ucirepo(id=186) 
    X = database.data.features 
    y = database.data.targets 
    y = pd.Series(np.ravel(y.quality>5)*1)
    return X, y

#--------------------------------------------------------
#6. RICE  | (3810x7)  | Binario

def rice():
    database = fetch_ucirepo(id=545) 
    X = database.data.features 
    y = database.data.targets 
    y = pd.Series((y['Class'] == 'Cammeo')*1)

    return X, y


#--------------------------------------------------------
#7. ABALONE | (4177x8)  | Binario (Era una regresión)

def abalone():
    database = fetch_ucirepo(id=1) 
    X = database.data.features 
    X.loc[:,'Sex'] = 1*(X['Sex'] == 'M')
    y = database.data.targets 
    y = pd.Series((y.Rings>9)*1)

    return X, y
'''
def wine_uci():
    df = pd.read_csv('data/wine_uci_186.csv', index_col = 0)
    X = df.drop('0', axis = 1)
    y = df['0']
    return X, y
#--------------------------------------------------------
#8. PEN-BASED | (10992x16)  | Multiclase 9 

def pen_based():
    col = np.append(['Att'+str(i) for i in range(0,16)], 'out')
    df_train = pd.read_csv('data/pendigits.tra', names = col)
    df_test = pd.read_csv('data/pendigits.tes', names = col)  
    df = pd.concat([df_train, df_test]).reset_index(drop = True)
    X = df.drop('out', axis = 1)
    y = df['out']
    y[y<5] = 0 
    y[y>=5] = 1 
    return X, y

#--------------------------------------------------------
#9. STATLOG LANDSAT | (6435 x 36)  | Binario adaptado (Era Multiclase 7)

def statlog_landsat():
    col = np.append(['Att'+str(i) for i in range(0,36)], 'out')
    df_train = pd.read_csv('data/UCI/statlog+landsat+satellite/sat.trn', names = col, sep = ' ')
    df_test = pd.read_csv('data/UCI/statlog+landsat+satellite/sat.tst', names = col, sep = ' ')
    df = pd.concat([df_train, df_test]).reset_index(drop = True)
    X = df.drop('out', axis = 1)
    y = df['out']
    y = pd.Series((y<4)*1)
    return X, y


#--------------------------------------------------------
#10. STATLOG SHUTTLE | (58000 x 8)  | Binario adaptado (Era Multiclase 7)

def statlog_shuttle():
    col = np.append(['Att'+str(i) for i in range(0,8)], 'out')
    df_train = pd.read_csv('data/shuttle.trn', names = col, sep = ' ')
    df_test = pd.read_csv('data/shuttle.tst', names = col, sep = ' ')
    df = pd.concat([df_train, df_test]).reset_index(drop = True)
    X = df.drop('out', axis = 1)
    y = df['out']
    y = pd.Series((y == 1)*1)
    return X, y


#--------------------------------------------------------
#11. OBESITY | (2111 x 16)  | Binario 
def obesity(from_csv=False):

    #Deprecated: load from the csv file    
    if from_csv:
        df = pd.read_csv('data/obesity.csv', sep = ',')
    else:
        # fetch dataset 
        #obesity_data = fetch_ucirepo(id=544) 
        
        # data (as pandas dataframes) 
        X = obesity_data.data.features 
        y = obesity_data.data.targets 
        
        df = pd.concat([X, y], axis=1)


    df['Gender'][df['Gender'] == 'Female'] = 0
    df['Gender'][df['Gender'] == 'Male'] = 1
    
    for i in ['FAVC', 'SMOKE', 'SCC', 'family_history_with_overweight']:
        df[i][df[i] == 'no'] = 0
        df[i][df[i] == 'yes'] = 1
        
    for i in ['CAEC', 'CALC']:
        df[i][df[i] == 'no'] = 0
        df[i][df[i] == 'Sometimes'] = 1
        df[i][df[i] == 'Frequently'] = 2
        df[i][df[i] == 'Always'] = 3
        
    for i in ['CAEC', 'CALC']:
        df[i][df[i] == 'no'] = 0
        df[i][df[i] == 'Sometimes'] = 1
        df[i][df[i] == 'Frequently'] = 2
        df[i][df[i] == 'Always'] = 3
        
    for count, j in enumerate(df['MTRANS'].unique()):
        df['MTRANS'][df['MTRANS'] == j] = count
    
    #OUTPUT
    for i in ['NObeyesdad']:
        df[i][df[i] == 'Insufficient_Weight'] = 1
        df[i][df[i] == 'Normal_Weight'] = 1
        df[i][df[i] == 'Overweight_Level_I'] = 1
        df[i][df[i] == 'Overweight_Level_II'] = 0
        df[i][df[i] == 'Obesity_Type_I'] = 0
        df[i][df[i] == 'Obesity_Type_II'] = 0
        df[i][df[i] == 'Obesity_Type_III'] = 0
        
    df_cat = df.select_dtypes(include=['object','category'])    
    for i in df_cat.columns:
        df[i] = pd.to_numeric(df[i])
        
        
    X = df.drop('NObeyesdad', axis = 1)
    y = df['NObeyesdad']    

    return X, y



#--------------------------------------------------------
#12. WHITE MATTER | (1904 x 24)  | Binario 


def wm():
    df = pd.read_csv('data/WM_data.csv', sep = ',')
    
    df.age5[df.age5 == '20-24'] = 1
    df.age5[df.age5 == '25-29'] = 2
    df.age5[df.age5 == '30-34'] = 3
    df.age5[df.age5 == '35-39'] = 4
    df.age5[df.age5 == '40-44'] = 5
    df.age5[df.age5 == '45-49'] = 6
    df.age5[df.age5 == '50-54'] = 7
    df.age5[df.age5 == '55-59'] = 8
    df.age5[df.age5 == '60-64'] = 9
    df.age5[df.age5 == '65-69'] = 10
    df.age5[df.age5 == '70-74'] = 11
    df.age5[df.age5 == '75-79'] = 12
    df.age5[df.age5 == '80-84'] = 13

    df.age5.astype(int)
    
    imp = df.drop('wm', axis = 1)
    index = []
    for i in range(imp.shape[1]):
        if len(imp.iloc[:,i].unique()) < 6:
            if imp.iloc[:,i].isnull().sum() != 0:
                index.append(i)

    imputer = KNNImputer(n_neighbors=5)
    i = imputer.fit_transform(imp)
    imp_correct = pd.DataFrame(i, columns = imp.columns)

    for i in index:
        for j in range(imp_correct.shape[0]):
            imp_correct.iloc[j, i] = round(imp_correct.iloc[j, i])
            
    X = imp_correct
    y = df.wm
    
    return X, y


#--------------------------------------------------------
#13. COLLINS | (500 x 22)  | Binario 
#Paper: Feature Selection in Learning Using Privileged Information
#url: https://www.openml.org/search?type=data&status=active&id=987
def collins():
    data, meta = arff.loadarff('data/fs_paper/collins.arff')
    df = pd.DataFrame(data)
    patron = r"'(.*?)'"
    df = df.drop('Text', axis = 1)
    df.Corpus = np.ravel([re.findall(patron, str(i)) for i in list(df.Corpus)])
    df.binaryClass = np.ravel([re.findall(patron, str(i)) for i in list(df.binaryClass)])
    X = df.drop('binaryClass', axis = 1)
    y = (df['binaryClass'] == 'P')*1
    
    return X, y 



#--------------------------------------------------------
#14. SYNTHETIC CONTROL | (600 x 6)  | Multiclase 6 
#Paper: Feature Selection in Learning Using Privileged Information
#url: https://www.openml.org/search?type=data&status=active&id=987
def synthetic_control():
    data, meta = arff.loadarff('data/fs_paper/synthetic_control.arff')
    df = pd.DataFrame(data)
    patron = r"'(.*?)'"
    df = df.drop('index', axis = 1)
    df['class'] = np.ravel([re.findall(patron, str(i)) for i in list(df['class'])])
    
    X = df.drop('class', axis = 1)
    y = df['class']
   
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    
    return X, y 





#--------------------------------------------------------
#15. MU 284 | (284 x 9) | Binario. Regresión/Multiclase en el paper.
#Paper: Feature Selection in Learning Using Privileged Information
#url: https://www.openml.org/search?type=data&status=active&id=987
def mu_284():
    data, meta = arff.loadarff('data/mu284.arff')
    df = pd.DataFrame(data)
    patron = r"'(.*?)'"
    df = df.drop('LABEL', axis = 1)
    df['binaryClass'] = np.ravel([re.findall(patron, str(i)) for i in list(df['binaryClass'])])
    X = df.drop('binaryClass', axis = 1)
    y = (df['binaryClass'] == 'P')*1
    
    return X, y 




#--------------------------------------------------------
#16. MAGICTELESCOPE | (19020x10) | Binario.
#Paper: Feature Selection in Learning Using Privileged Information
#url: https://www.openml.org/search?type=data&status=active&id=987
def magictelescope():
    data, meta = arff.loadarff('data/MagicTelescope.arff')
    df = pd.DataFrame(data)
    patron = r"'(.*?)'"
    df = df.drop('ID', axis = 1)
    df['class:'] = np.ravel([re.findall(patron, str(i)) for i in list(df['class:'])])
    X = df.drop('class:', axis = 1)
    y = (df['class:'] == 'g')*1

    return X, y 




#--------------------------------------------------------
#17. POLLEN | (3848x5) | Binario.
#Paper: Feature Selection in Learning Using Privileged Information
#url: https://www.openml.org/search?type=data&status=active&id=987
def pollen():
    data, meta = arff.loadarff('data/pollen.arff')
    df = pd.DataFrame(data)
    patron = r"'(.*?)'"
    df['binaryClass'] = np.ravel([re.findall(patron, str(i)) for i in list(df['binaryClass'])])
    X = df.drop('binaryClass', axis = 1)
    y = (df['binaryClass'] == 'P')*1
    return X, y 

#--------------------------------------------------------
#18. WILT | (4839x5) | Binario.
#Paper: Feature Selection in Learning Using Privileged Information
#url: https://www.openml.org/search?type=data&status=active&id=987
def wilt():
    data, meta = arff.loadarff('data/wilt.arff')
    df = pd.DataFrame(data)
    patron = r"'(.*?)'"
    df['Class'] = np.ravel([re.findall(patron, str(i)) for i in list(df['Class'])])
    X = df.drop('Class', axis = 1)
    y = (df['Class'] == '1')*1
    return X, y 

#--------------------------------------------------------
#19. ADA AGNOSTIC | (4562x48) | Binario.
#Paper: Feature Selection in Learning Using Privileged Information
#url: https://www.openml.org/search?type=data&status=active&id=987
def ada_agnostic():
    data, meta = arff.loadarff('data/ada_agnostic.arff')
    df = pd.DataFrame(data)
    patron = r"'(.*?)'"
    df['label'] = np.ravel([re.findall(patron, str(i)) for i in list(df['label'])])
    X = df.drop(['label', 'attr39'], axis = 1)
    y = (df['label'] == '1' )*1
    return X, y 


#--------------------------------------------------------
#20. kc1 | (2109x21) | Binario.
#Paper: Feature Selection in Learning Using Privileged Information
#url: https://www.openml.org/search?type=data&status=active&id=987
def kc1():
    data, meta = arff.loadarff('data/kc1.arff')
    df = pd.DataFrame(data)
    patron = r"'(.*?)'"
    df['defects'] = np.ravel([re.findall(patron, str(i)) for i in list(df['defects'])])
    X = df.drop('defects', axis = 1)
    y = (df['defects'] == 'true' )*1
    return X, y 

#--------------------------------------------------------
#21. Phoneme | (5404x5) | Binario.
#url: https://www.openml.org/search?type=data&status=any&id=1489
def phoneme():
    data, meta = arff.loadarff('data/phoneme.arff')
    df = pd.DataFrame(data)
    patron = r"'(.*?)'"
    df['Class'] = np.ravel([re.findall(patron, str(i)) for i in list(df['Class'])])
    X = df.drop('Class', axis = 1)
    y = (df['Class'] == '1' )*1
    return X, y 

#--------------------------------------------------------
#22. Mozilla4 | (15545x5) | Binario.
#CUIDADO COLUMNA ID
#url: https://www.openml.org/search?type=data&status=active&id=987
def mozilla4():
    data, meta = arff.loadarff('data/mozilla4.arff')
    df = pd.DataFrame(data)
    patron = r"'(.*?)'"
    df['state'] = np.ravel([re.findall(patron, str(i)) for i in list(df['state'])])
    X = df.drop('state', axis = 1)
    y = (df['state'] == '1' )*1
    return X, y 

#--------------------------------------------------------
#23. Elect | (15545x5) | Binario.
#url: https://www.openml.org/search?type=data&status=active&id=987
def elect():
    data, meta = arff.loadarff('data/electricity-normalized.arff')
    df = pd.DataFrame(data)
    df = df.drop(['date', 'day'], axis = 1)
    patron = r"'(.*?)'"
    df['class'] = np.ravel([re.findall(patron, str(i)) for i in list(df['class'])])
    X = df.drop('class', axis = 1)
    y = (df['class'] == 'UP' )*1
    return X, y 


#--------------------------------------------------------
#24. Phishing | (11055X30) | Binario.
#url: https://www.openml.org/search?type=data&status=active&id=987
def phishing_big():
    data, meta = arff.loadarff('data/phishing_big.arff')
    df = pd.DataFrame(data)
    patron = r"'(.*?)'"
    for j in df.columns:
        df[j] = np.ravel([re.findall(patron, str(i)) for i in list(df[j])])
        df[j] = pd.to_numeric(df[j])
    X = df.drop('Result', axis = 1)
    y = (df['Result'] == 1 )*1
    return X, y 


#--------------------------------------------------------
#25. MNIST (2x2) 4 y 9 
def mnist2x2():
    X_train = pd.read_csv('data/mnist/X_train2_49.csv', index_col = 0)
    X_test = pd.read_csv('data/mnist/X_test2_49.csv', index_col = 0)
    y_test = pd.read_csv('data/mnist/y_test49.csv', index_col = 0)
    y_train = pd.read_csv('data/mnist/y_train49.csv', index_col = 0)
    
    X = pd.concat([X_train, X_test]).reset_index(drop = True)
    y = pd.concat([y_train, y_test]).reset_index(drop = True)['0']
    
    return X, y


#--------------------------------------------------------
#26. MNIST (3x3) 4 y 9 
def mnist3x3():
    X_train = pd.read_csv('data/mnist/X_train3_49.csv', index_col = 0)
    X_test = pd.read_csv('data/mnist/X_test3_49.csv', index_col = 0)
    y_test = pd.read_csv('data/mnist/y_test49.csv', index_col = 0)
    y_train = pd.read_csv('data/mnist/y_train49.csv', index_col = 0)
    
    X = pd.concat([X_train, X_test]).reset_index(drop = True)
    y = pd.concat([y_train, y_test]).reset_index(drop = True)['0']
    
    return X, y



'''
#--------------------------------------------------------
#27. IONOSPHERE
def ionosphere():
    ionosphere = fetch_ucirepo(id=52) 
    X = ionosphere.data.features 
    X = X.drop('Attribute2', axis = 1)
    y = ionosphere.data.targets['Class']
    y = pd.Series(list((y == 'g')*1))
    pi_features = ['Attribute5', 'Attribute6', 'Attribute21', 'Attribute22']
    
    return X, y, pi_features
'''

#--------------------------------------------------------
#28. kc2
def kc2():
    dfs = pd.read_csv('data/PI/kc2.csv')
    
    dfs['problems'] = (dfs['problems'] == 'yes')*1
    X = dfs.drop('problems', axis = 1)
    y = dfs['problems']
    pi_features= list(X.columns[14:])
    
    return X, y, pi_features

#--------------------------------------------------------
#29. Parkinsons
def parkinsons():
    df = pd.read_csv('data/PI/parkinsons/parkinsons.data')
    X = df.drop(['name', 'status'], axis = 1)
    y = df['status']
    
    col = X.columns
    scaler = StandardScaler()
    Xnorm = scaler.fit_transform(X)
    Xn = pd.DataFrame(Xnorm, columns = col)
    
    
    
    
    
    mi = mutual_info_classif(Xn, y)
    mi_df = pd.DataFrame({'name': list(Xn.columns), 'mi': mi })
    mi_sort = mi_df.sort_values(by='mi', ascending=False)
    pi_features = list(mi_sort['name'][0:])[0:10]
    return X, y, pi_features



#--------------------------------------------------------
#30. MNIST
def mnist_r():
    X_28 = pd.read_csv('data/mnist/X_14.csv', index_col = [0])
    X_5  = pd.read_csv('data/mnist/X_7.csv', index_col = [0])
    y =  pd.read_csv('data/mnist/y.csv', index_col = [0]).reset_index(drop = True).iloc[:,0]
    X_5.columns = ['p' + str(col) for col in X_5.columns]
    pi_features = X_28.columns 
    
    X = pd.concat([X_28, X_5], axis = 1).reset_index(drop = True)


    return X, y, pi_features




def mnist_g():
    X_28 = pd.read_csv('data/mnist/gX_14.csv', index_col = [0])
    X_5  = pd.read_csv('data/mnist/gX_14G.csv', index_col = [0])
    y =  pd.read_csv('data/mnist/gy.csv', index_col = [0]).reset_index(drop = True).iloc[:,0]
    X_5.columns = ['p' + str(col) for col in X_5.columns]
    pi_features = X_28.columns 
    
    X = pd.concat([X_28, X_5], axis = 1).reset_index(drop = True)


    return X, y, pi_features





def cifar10():
    X_pi = pd.read_csv('data/cifar10_19/X_pi.csv', index_col = [0])
    X_reg  = pd.read_csv('data/cifar10_19/X_reg.csv', index_col = [0])
    y = pd.read_csv('data/cifar10_19/y.csv', index_col = [0]).iloc[:,0]
    X_reg.columns = ['p' + str(col) for col in X_reg.columns]
    pi_features = X_pi.columns
    
    X = pd.concat([X_pi, X_reg], axis = 1).reset_index(drop = True)
    
    return X, y, pi_features


def fruit():
    X_pi = pd.read_csv('data/fruit/X_14.csv', index_col = [0])
    X_reg  = pd.read_csv('data/fruit/X_7.csv', index_col = [0])
    y = pd.read_csv('data/fruit/y.csv', index_col = [0]).iloc[:,0]
    X_reg.columns = ['p' + str(col) for col in X_reg.columns]
    pi_features = X_pi.columns
    
    X = pd.concat([X_pi, X_reg], axis = 1).reset_index(drop = True)
    
    return X, y, pi_features


def fashion():
    df = pd.read_csv('data/fashion.csv', index_col = [0])
    regular = ['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'year', 'usage']
    pi_features = [i for i in df.columns if i not in regular ]
    X = df.drop('usage', axis = 1)
    y = df['usage']
    return X, y, pi_features


def hds():
    X_pi = pd.read_csv('data/HDS_or/X_10.csv', index_col = [0])
    X_reg  = pd.read_csv('data/HDS_or/X_5.csv', index_col = [0])
    y = pd.read_csv('data/HDS_or/y.csv', index_col = [0]).iloc[:,0]
    X_reg.columns = ['p' + str(col) for col in X_reg.columns]
    pi_features = X_pi.columns
    
    X = pd.concat([X_pi, X_reg], axis = 1).reset_index(drop = True)
    
    return X, y, pi_features
    
    
#------------------------------
#NUEVA RONDA 5/11/24
#------------------------------

def kin():
    data, meta = arff.loadarff('data/kin8nm.arff')
    df = pd.DataFrame(data)
    patron = r"'(.*?)'"
    df.binaryClass = np.ravel([re.findall(patron, str(i)) for i in list(df.binaryClass)])
    label_encoder = LabelEncoder()
    y = pd.Series(label_encoder.fit_transform(df['binaryClass']))
    X = df.drop('binaryClass', axis = 1)
    return X, y 


def elevators():
    data, meta = arff.loadarff('data/delta_elevators.arff')
    df = pd.DataFrame(data)
    patron = r"'(.*?)'"
    df.binaryClass = np.ravel([re.findall(patron, str(i)) for i in list(df.binaryClass)])
    label_encoder = LabelEncoder()
    y = pd.Series(label_encoder.fit_transform(df['binaryClass']))
    X = df.drop('binaryClass', axis = 1)
    return X, y 

def visual():
    data, meta = arff.loadarff('data/visualizing_soil.arff')
    df = pd.DataFrame(data)
    patron = r"'(.*?)'"
    df['binaryClass'] = np.ravel([re.findall(patron, str(i)) for i in list(df['binaryClass'])])
    df['isns'] = np.ravel([re.findall(patron, str(i)) for i in list(df['isns'])])
    label_encoder = LabelEncoder()
    y = pd.Series(label_encoder.fit_transform(df['binaryClass']))
    X = df.drop('binaryClass', axis = 1)
    return X,y

def abal():
    data, meta = arff.loadarff('data/abalone.arff')
    df = pd.DataFrame(data)
    patron = r"'(.*?)'"
    df['binaryClass'] = np.ravel([re.findall(patron, str(i)) for i in list(df['binaryClass'])])
    df['Sex'] = np.ravel([re.findall(patron, str(i)) for i in list(df['Sex'])])
    label_encoder = LabelEncoder()
    y = pd.Series(label_encoder.fit_transform(df['binaryClass']))
    df['Sex'] = label_encoder.fit_transform(df['Sex'])
    X = df.drop('binaryClass', axis = 1)
    return X, y

def wind():
    data, meta = arff.loadarff('data/wind.arff')
    df = pd.DataFrame(data)
    patron = r"'(.*?)'"
    df.binaryClass = np.ravel([re.findall(patron, str(i)) for i in list(df.binaryClass)])
    label_encoder = LabelEncoder()
    y = pd.Series(label_encoder.fit_transform(df['binaryClass']))
    X = df.drop('binaryClass', axis = 1)
    return X, y


def ring_norm():
    data, meta = arff.loadarff('data/phpWfYmlu.arff')
    df = pd.DataFrame(data)
    patron = r"'(.*?)'"
    df.Class = np.ravel([re.findall(patron, str(i)) for i in list(df.Class)])
    label_encoder = LabelEncoder()
    y = pd.Series(label_encoder.fit_transform(df['Class']))
    X = df.drop('Class', axis = 1)
    return X, y


def sil():
    data, meta = arff.loadarff('data/file7a97574fa9ae.arff')
    df = pd.DataFrame(data)
    patron = r"'(.*?)'"
    df['class'] = np.ravel([re.findall(patron, str(i)) for i in list(df['class'])])
    label_encoder = LabelEncoder()
    y = pd.Series(label_encoder.fit_transform(df['class']))
    X = df.drop('class', axis = 1)
    return X, y
  

'''
  text = ['phishing', 'obesity', 'diabetes', 'wm', 'phoneme', 'magic_telescope', 'mozilla', 'elect', 'wine', 'kin', 'elevators', 'abalone', 'wind', 'sil']

sam = []
feat = []
y1 =[]
y0 = []

for ind in text:
    X, y = datasets_dict[ind]
    sam.append(X.shape[0])
    feat.append(X.shape[1])
    y1.append(np.round((len(y[y==1])/len(y))*100,1))
    y0.append(np.round((len(y[y==0])/len(y))*100,1))


off = pd.DataFrame({'Dataset':text, 'Samples': sam, 'Columms': feat, 'Class1': y1, 'Class0': y0})
  
  
  
'''