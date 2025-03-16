# MORE ABOUT THIS FILE:

# Main file for the tabular results presented in the paper.
# One privileged feature


# %%
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error
import datasets as dat
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
import utils as ut
import models as mo
import scipy.optimize as so
import argparse
from tensorflow import keras
from tensorflow.keras import layers

#python MT_main.py -dataset obesity  -epo 2 -bs 128 -vs 0.2 -pat 1 -iter 1


parser = argparse.ArgumentParser()
# Define arguments
parser.add_argument("-dataset", dest = "dataset", nargs = '+' )
parser.add_argument("-l_reg", dest = "l_reg", nargs = '+' , default = [], type = int)
parser.add_argument("-l_clas", dest = "l_clas", nargs = '+', default = [], type = int)
parser.add_argument("-epo", dest = "epo", type = int)
parser.add_argument("-bs", dest = "bs", type = int)
parser.add_argument("-vs", dest = "vs", type = float)
parser.add_argument("-pat", dest = "pat", type = int)
parser.add_argument("-iter", dest = "iter", type = int)
args = parser.parse_args()


# %%
text    = ['phishing', 'obesity', 'diabetes', 'phoneme', 'mozilla',\
            'wine', 'abalone', 'wind'] 
dataset = [ dat.phishing(from_csv = True), dat.obesity(from_csv = True), dat.diabetes(),\
            dat.phoneme(),  dat.mozilla4(),  dat.wine_uci(), \
            dat.abal(), dat.wind()]

datasets_dict = dict(zip(text, dataset))


# NN architectures of the regressor and the classifier
lay_reg  = args.l_reg      #Layers for the regressor
lay_clas = args.l_clas      #Layers for the classifier
epo = args.epo     #Epochs 100
bs = args.bs       #Batch Size 32
vs = args.vs           #Validation Split
es = True
pat = args.pat            #Patience

# Determines the number of iterations of the k-fold CV
n_iter  = args.iter
ran = np.random.randint(1000, size = n_iter)

dff = pd.DataFrame()  #Dataframe to store the results of each dataset

# %%

'''
text = ['phishing']
drp = False
epo = 2
bs = 128
vs = 0.2
pat = 1
val_drp = 0
n_iter = 1
ran = np.random.randint(1000, size = n_iter)

lay_reg = []
lay_clas = []
'''

# %%

#Process each dataset
for ind in args.dataset:
    t = ind #text of the current dataset

    #Retrieve the current dataset and extract the privileged feature
    if t in [ 'mnist_r', 'fruit', 'mnist_g']:
        X, y, pi_features = datasets_dict[ind]
        X = X/255
        shap = len(pi_features)
    else:
        X, y = datasets_dict[ind]
        pi_features = ut.feat_correlation(X,y)
        shap = 1

    X = X.sample(frac = 1)
    ind = X.index
    X = X.reset_index(drop = True)
    y = y[ind].reset_index(drop = True)

    #Number of folds
    cv = 5
    print(t)
    #Create a list to save the results 
    err_up_pri, err_up, err_b, err_pfd, err_gd, err_tpd = [[] for i in range(6)]
    err1, err1PFD, err1TPD = [[] for i in range(3)]
    err2, err2PFD, err2TPD = [[] for i in range(3)]

    err1TPDi, mae1TPDi = [[] for i in range(2)]
    err2TPDi, mae2TPDi = [[] for i in range(2)]


    mae1, mae1PFD, mae1TPD  = [[] for i in range(3)]
    mae2, mae2PFD, mae2TPD = [[] for i in range(3)]

    err_nn, err_nnPFD, err_nnTPD, mae_nn = [[] for i in range(4)]

    per, std_per, dist = [[] for i in range(3)]
    #For each fold (k is the random seed of each fold)
    for k in ran:
        #Create a dictionary with all the fold partitions
        dr = ut.skfold(X, pd.Series(list(y)), cv, r = k, name = 'stratified')
        #Process each fold individually
        for h in range(cv):
            #Get the current partition
            X_train, y_train, X_test, y_test  = ut.train_test_fold(dr, h)
            
            #Preprocess the data
            SS = StandardScaler()
            if t not in ['elect']:
                X_train = pd.DataFrame(SS.fit_transform(X_train), columns = X_train.columns)
                X_test = pd.DataFrame(SS.transform(X_test), columns = X_train.columns)
        
            # Get the privileged feature
            pri = X_train[pi_features]
            pri_test = X_test[pi_features]
    
            #Drop the privileged feature from the train set
            X_trainr = X_train.drop(pi_features, axis = 1)
            X_testr = X_test.drop(pi_features, axis = 1)
            
            
            #----------------------------------------------------------
            #UPPER (REGULAR + PRIV)
            #----------------------------------------------------------

            #Create the model 
            model =  mo.nn_binary_clasification( X_train.shape[1], lay_clas, 'relu')   
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            #Fit the model
            model.fit(X_train, y_train, epochs=epo, batch_size=bs, verbose = 0, validation_split = vs, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = pat )])

            #Measure test error
            y_pre_up = np.ravel([np.round(i) for i in model.predict(X_test)])
            err_up.append(1-accuracy_score(y_test, y_pre_up))
            y_pre_prob = model.predict(X_train)
            delta_i = np.array((y_train == np.round(np.ravel(y_pre_prob)))*1)
    

            #----------------------------------------------------------
            #LOWER
            #----------------------------------------------------------

            model =  mo.nn_binary_clasification( X_trainr.shape[1], lay_clas, 'relu')
            model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
            model.fit(X_trainr, y_train, epochs=epo, batch_size=bs, verbose = 0, validation_split = vs, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = pat)])

            #Measure test error
            y_pre_b = np.ravel([np.round(i) for i in model.predict(X_testr)])

            err_b.append(1-accuracy_score(y_test, y_pre_b))

    

            #XXXXXXXXXXXXXXXXXXXXXXX
            #DISTILLATION APPROACHES
            #XXXXXXXXXXXXXXXXXXXXXXX

            y_MT = np.column_stack([np.ravel(y_train), np.ravel(pri), np.ravel(y_pre_prob), np.ravel(delta_i)])

            #---------------------------------------------------------
            #PFD
            #----------------------------------------------------------
            yy_PFD = np.column_stack([np.ravel(y_train), np.ravel(y_pre_prob)])
            
            model =  mo.nn_binary_clasification( X_trainr.shape[1], lay_clas, 'relu')     
            model.compile(loss= ut.loss_GD(1, 0.5), optimizer='adam', metrics=['accuracy'])
            model.fit(X_trainr, yy_PFD, epochs=epo, batch_size=bs, verbose = 0, validation_split = vs, 
                      callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = pat)])

            
            #Measure test error
            y_pre = np.ravel([np.round(i) for i in model.predict(X_testr)])
            err_pfd.append(1-accuracy_score(y_test, y_pre))  



            #---------------------------------------------------------
            #TPD
            #----------------------------------------------------------
            yy_TPD = np.column_stack([np.ravel(y_train), np.ravel(y_pre_prob), delta_i])
    
    
            model =  mo.nn_binary_clasification( X_trainr.shape[1], lay_clas, 'relu')     
            model.compile(loss= ut.loss_TPD(1, 1, 0.5), optimizer='adam', metrics=['accuracy'])
   
            #Fit the model
            model.fit(X_trainr, yy_TPD, epochs=epo, batch_size=bs, verbose = 0, validation_split = vs,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = pat)])      


            #Measure test error
            y_pre = np.ravel([np.round(i) for i in model.predict(X_testr)])
            err_tpd.append(1-accuracy_score(y_test, y_pre))

            #XXXXXXXXXXXXXXXXXXXXXXX
            #MULTI-TASK
            #XXXXXXXXXXXXXXXXXXXXXXX   

            #----------------------------------------------------------
            #MTP
            #----------------------------------------------------------

            
            model  = mo.mt2(X_trainr.shape[1], lay_clas, lay_reg, 'relu')
            model.compile(loss=ut.multi_task, optimizer="adam", metrics=['accuracy'])
            model.fit(X_trainr, y_MT, epochs=epo, batch_size=bs, verbose = 0, validation_split = vs, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = pat)])
            p = model.predict(X_testr)
            err2.append(1-accuracy_score(y_test, np.round(p[:,0])))
            mae2.append(mean_absolute_error(pri_test, p[:,1]))

            #----------------------------------------------------------
            #MTP-PFD
            #----------------------------------------------------------

            model  = mo.mt2(X_trainr.shape[1], lay_clas, lay_reg, 'relu')
            model.compile(loss=ut.multi_taskPFD, optimizer="adam", metrics=['accuracy'])
            model.fit(X_trainr, y_MT, epochs=epo, batch_size=bs, verbose = 0, validation_split = vs, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = pat)])
            p = model.predict(X_testr)
            err2PFD.append(1-accuracy_score(y_test, np.round(p[:,0])))
            mae2PFD.append(mean_absolute_error(pri_test, p[:,1]))

            #----------------------------------------------------------
            #MTP-TPD
            #----------------------------------------------------------

            model  = mo.mt2(X_trainr.shape[1], lay_clas, lay_reg, 'relu')
            model.compile(loss=ut.multi_taskTPD, optimizer="adam", metrics=['accuracy'])
            model.fit(X_trainr, y_MT, epochs=epo, batch_size=bs, verbose = 0, validation_split = vs, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = pat)])
            p = model.predict(X_testr)
            err2TPD.append(1-accuracy_score(y_test, np.round(p[:,0])))
            mae2TPD.append(mean_absolute_error(pri_test, p[:,1]))


            #XXXXXXXXXXXXXXXXXXXXXXX
            #KNOWLEDGE-TRANSFER
            #XXXXXXXXXXXXXXXXXXXXXXX  


            model_reg = mo.nn_reg1(X_trainr.shape[1], lay_reg, 'relu')
            model_reg.compile(loss="mean_squared_error", optimizer="adam", metrics=['mean_absolute_error'])
            model_reg.fit(X_trainr, pri, epochs=epo, batch_size=bs, verbose = 0, validation_split = vs, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = pat)])

            pri_nn_train = np.ravel(model_reg.predict(X_trainr))
            pri_nn_test = np.ravel(model_reg.predict(X_testr))

            mae_nn.append(mean_absolute_error(pri_test, pri_nn_test))

            X_trainr.insert(loc = X_train.columns.get_loc(pi_features), column = 'pri', value = pri_nn_train)
            X_testr.insert(loc = X_train.columns.get_loc(pi_features), column = 'pri', value = pri_nn_test)
            X_trainr.columns = X_trainr.columns.astype(str)
            X_testr.columns = X_testr.columns.astype(str)

            #----------------------------------------------------------
            #ESTANDAR
            #----------------------------------------------------------

            model =  mo.nn_binary_clasification( X_trainr.shape[1], lay_clas, 'relu')
            model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
            model.fit(X_trainr, y_train, epochs=epo, batch_size=bs, verbose = 0, validation_split = vs, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = pat)])

            y_pre = model.predict(X_testr, verbose = 0)
            err_nn.append(1-accuracy_score(y_test, np.ravel(np.round(y_pre))))

            #----------------------------------------------------------
            #PFD
            #----------------------------------------------------------

            yy_PFD = np.column_stack([np.ravel(y_train), np.ravel(y_pre_prob)])
            model =  mo.nn_binary_clasification( X_trainr.shape[1], lay_clas, 'relu')
            model.compile(loss=ut.loss_GD(1, 0.5), optimizer="adam", metrics=['accuracy'])
            model.fit(X_trainr, yy_PFD, epochs=epo, batch_size=bs, verbose = 0, validation_split = vs, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = pat)])

            y_pre = model.predict(X_testr, verbose = 0)
            err_nnPFD.append(1-accuracy_score(y_test, np.ravel(np.round(y_pre))))

            
            #----------------------------------------------------------
            #TPD
            #----------------------------------------------------------

            yy_TPD = np.column_stack([np.ravel(y_train), np.ravel(y_pre_prob), delta_i])
            model =  mo.nn_binary_clasification( X_trainr.shape[1], lay_clas, 'relu')
            model.compile(loss= ut.loss_TPD(1, 1, 0.5), optimizer="adam", metrics=['accuracy'])
            model.fit(X_trainr, yy_TPD, epochs=epo, batch_size=bs, verbose = 0, validation_split = vs, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = pat)])

            y_pre = model.predict(X_testr, verbose = 0)
            err_nnTPD.append(1-accuracy_score(y_test, np.ravel(np.round(y_pre))))

            tf.keras.backend.clear_session()
    #Save the results
    off = {'name' : t,
            'up': np.round(np.mean(err_up), 3),
            'b': np.round(np.mean(err_b), 3),
            'KT': np.round(np.mean(err_nn), 3),
            'KT-PFD': np.round(np.mean(err_nnPFD), 3),
            'KT-TPD': np.round(np.mean(err_nnTPD), 3),
            'MT': np.round(np.mean(err2), 3),
            'MT-PFD': np.round(np.mean(err2PFD), 3),
            'MT-TPD': np.round(np.mean(err2TPD), 3),
            'PFD': np.round(np.mean(err_pfd), 3),
            'TPD': np.round(np.mean(err_tpd), 3),
            'std_up': np.round(np.std(err_up), 3),
            'std_b': np.round(np.std(err_b), 3),
            'std_KT': np.round(np.std(err_nn), 3),
            'std_KT-PFD': np.round(np.std(err_nnPFD), 3),
            'std_KT-TPD': np.round(np.std(err_nnTPD), 3),
            'std_MT': np.round(np.std(err2), 3),
            'std_MT-PFD': np.round(np.std(err2PFD), 3),
            'std_MT-TPD': np.round(np.std(err2TPD), 3),
            'std_PFD': np.round(np.std(err_pfd), 3),
            'std_TPD': np.round(np.std(err_tpd), 3),
            'LUPI_KT %': ut.LUPI_gain(np.round(np.mean(err_up), 3), np.round(np.mean(err_b), 3), np.round(np.mean(err_nn), 3)),
            'LUPI_KT-PFD %': ut.LUPI_gain(np.round(np.mean(err_up), 3), np.round(np.mean(err_b), 3), np.round(np.mean(err_nnPFD), 3)),
            'LUPI_KT-TPD%': ut.LUPI_gain(np.round(np.mean(err_up), 3), np.round(np.mean(err_b), 3), np.round(np.mean(err_nnTPD), 3)),
            'LUPI_MT %': ut.LUPI_gain(np.round(np.mean(err_up), 3), np.round(np.mean(err_b), 3), np.round(np.mean(err2), 3)),
            'LUPI_MT-PFD %': ut.LUPI_gain(np.round(np.mean(err_up), 3), np.round(np.mean(err_b), 3), np.round(np.mean(err2PFD), 3)),
            'LUPI_MT-TPD %': ut.LUPI_gain(np.round(np.mean(err_up), 3), np.round(np.mean(err_b), 3), np.round(np.mean(err2TPD), 3)),
            'LUPI_PFD %': ut.LUPI_gain(np.round(np.mean(err_up), 3), np.round(np.mean(err_b), 3), np.round(np.mean(err_pfd), 3)),
            'LUPI_TPD %': ut.LUPI_gain(np.round(np.mean(err_up), 3), np.round(np.mean(err_b), 3), np.round(np.mean(err_tpd), 3)),
            'mae_KT': np.round(np.mean(mae_nn), 3),
            'mae_MT': np.round(np.mean(mae2), 3),
            'mae_MT-PFD': np.round(np.mean(mae2PFD), 3),
            'mae_MT-TPD': np.round(np.mean(mae2TPD), 3)
           }   
    
 
    df1 = pd.DataFrame(off, index = [0])
        
    dff  = pd.concat([dff, df1]).reset_index(drop = True)


v = '_'.join(args.dataset)
str_reg = [str(i) for i in args.l_reg]
lr = '-'.join(str_reg)
str_clas = [str(i) for i in args.l_clas]
lc = '-'.join(str_clas)

dff.to_csv('MT_' + v + '_' + lr + '_' + lc + '_' + str(epo) + '_' + str(bs) + '_' + str(vs) + '_' + str(pat)+ '_' + str(n_iter)+ '.csv')
    
# %%
