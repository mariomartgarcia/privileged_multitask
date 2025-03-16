# MORE ABOUT THIS FILE:

# This file introduced the EPSILON parameter of the paper. 
# The results to design the plots are obtained from here.



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
#python MT_weight.py -dataset obesity -w 0.25 0.5 0.75 -epo 2 -bs 128 -vs 0.2 -pat 1 -iter 1
#python MT_weight.py -dataset obesity -w 1 2 -epo 2 -bs 128 -vs 0.2 -pat 1 -iter 1
#python MT_weight.py -dataset sil kin elect wm -w 1 2 5 10 20 -epo 1000 -bs 128 -vs 0.2 -pat 5 -iter 20

parser = argparse.ArgumentParser()
# Define arguments
parser.add_argument("-dataset", dest = "dataset", nargs = '+' )
parser.add_argument("-l_reg", dest = "l_reg", nargs = '+' , default = [], type = int)
parser.add_argument("-l_clas", dest = "l_clas", nargs = '+', default = [], type = int)
parser.add_argument("-w", dest = "w", nargs = '+', default = [0.5], type = float)
parser.add_argument("-epo", dest = "epo", type = int)
parser.add_argument("-bs", dest = "bs", type = int)
parser.add_argument("-vs", dest = "vs", type = float)
parser.add_argument("-pat", dest = "pat", type = int)
parser.add_argument("-iter", dest = "iter", type = int)
args = parser.parse_args()


# %%
text    = ['phishing', 'obesity', 'diabetes', 'wm', 'phoneme', 'magic_telescope', 'mozilla', 'mnist_r',\
            'fruit', 'mnist_g', 'elect', 'wine', 'kin', 'elevators', 'visual', 'abalone', 'wind', 'sil'] 
dataset = [ dat.phishing(from_csv = True), dat.obesity(from_csv = True), dat.diabetes(), dat.wm() ,\
            dat.phoneme(), dat.magictelescope(),  dat.mozilla4(), dat.mnist_r(), \
            dat.fruit(), dat.mnist_g(), dat.elect(), dat.wine_uci(), \
            dat.kin(), dat.elevators(), dat.visual(), dat.abal(), \
            dat.wind(), dat.sil()]

datasets_dict = dict(zip(text, dataset))


# NN architectures of the regressor and the classifier
lay_reg  = args.l_reg      #Layers for the regressor
lay_clas = args.l_clas      #Layers for the classifier
epo = args.epo     #Epochs 100
bs = args.bs       #Batch Size 32
vs = args.vs           #Validation Split
pat = args.pat            #Patience
omega = args.w

# Determines the number of iterations of the k-fold CV
n_iter  = args.iter
ran = np.random.randint(1000, size = n_iter)

dff = pd.DataFrame()  #Dataframe to store the results of each dataset

# %%
'''
text = ['obesity']
drp = False
epo = 2
bs = 128
vs = 0.2
pat = 1
val_drp = 0
n_iter = 2
ran = np.random.randint(1000, size = n_iter)
omega = [1,2]
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

    err_nn, err_nnPFD, err_nnTPD, mae_nn = [[] for i in range(4)]
    dfw = pd.DataFrame([])

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
            
            
            #TRAIN THE THREE MAIN MODELS: UPPER, LOWER AND PRIVILEGED
            ###########################################################
            #----------------------------------------------------------
            #UPPER (PRIV)
            #----------------------------------------------------------

            #Create the model 
            model =  mo.nn_binary_clasification( shap, lay_clas, 'relu')   
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            #Fit the model
            model.fit(pri, y_train, epochs=epo, batch_size=bs, verbose = 0, validation_split = vs, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = pat )])

            #Measure test error
            y_pre_up = np.ravel([np.round(i) for i in model.predict(pri_test)])
            err_up_pri.append(1-accuracy_score(y_test, y_pre_up))
            y_pre_probpi = model.predict(pri)

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
                
            #Rper, Rdist, Rstd_per = ut.metrics(pri_test, X_train, X_testr, model, pi_features)
            #X_testr = X_testr.drop(pi_features, axis = 1)

            #per.extend(Rper)
            #dist.extend(Rdist)
            #std_per.extend(Rstd_per)

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

            '''
            #---------------------------------------------------------
            #GD
            #----------------------------------------------------------

            
            yy_GD = np.column_stack([np.ravel(y_train), np.ravel(y_pre_probpi)])
            
            model =  mo.nn_binary_clasification( X_trainr.shape[1], lay_clas, 'relu')     
            model.compile(loss= ut.loss_GD(1, 0.5), optimizer='adam', metrics=['accuracy'])
            model.fit(X_trainr, yy_GD, epochs=epo, batch_size=bs, verbose = 0, validation_split = vs, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = pat)])

            
            #Measure test error
            y_pre = np.ravel([np.round(i) for i in model.predict(X_testr)])
            err_gd.append(1-accuracy_score(y_test, y_pre))  


            #---------------------------------------------------------
            #PFD
            #----------------------------------------------------------

            
            yy_PFD = np.column_stack([np.ravel(y_train), np.ravel(y_pre_prob)])
            
            model =  mo.nn_binary_clasification( X_trainr.shape[1], lay_clas, 'relu')     
            model.compile(loss= ut.loss_GD(1, 0.5), optimizer='adam', metrics=['accuracy'])
            model.fit(X_trainr, yy_PFD, epochs=epo, batch_size=bs, verbose = 0, validation_split = vs, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = pat)])

            
            #Measure test error
            y_pre = np.ravel([np.round(i) for i in model.predict(X_testr)])
            err_pfd.append(1-accuracy_score(y_test, y_pre))  



            #---------------------------------------------------------
            #TPD
            #----------------------------------------------------------

            delta_i = np.array((y_train == np.round(np.ravel(y_pre_prob)))*1)
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

            X_trainr = X_trainr.drop('pri', axis = 1)
            X_testr = X_testr.drop('pri', axis = 1)
            '''


            #XXXXXXXXXXXXXXXXXXXXXXX
            #MULTI-TASKS
            #XXXXXXXXXXXXXXXXXXXXXXX   

            #---------------------------------------------------------
            #---------------------------------------------------------

            #MODEL1: Standard Multi-task


            for ww in omega:
                
                green, red, purple, yellow = [[] for i in range(4)]
                greenb, redb, purpleb, yellowb = [[] for i in range(4)]

                com, comb = [[] for i in range(2)]
                
                '''
                #----------------------------------------------------------
                #NORMAL M1
                #----------------------------------------------------------

                model = mo.mt1(X_trainr.shape[1], lay_clas, 'relu')
                model.compile(loss=ut.Wmulti_task(ww), optimizer="adam", metrics=['accuracy'])
                model.fit(X_trainr, y_MT, epochs=epo, batch_size=bs, verbose = 0, validation_split = vs, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = pat )])
                p = model.predict(X_testr)
                err1 = (1-accuracy_score(y_test, np.round(p[:,0])))
                mae1 = (mean_absolute_error(pri_test, p[:,1]))

                #----------------------------------------------------------
                #PFD M1
                #----------------------------------------------------------

                model = mo.mt1(X_trainr.shape[1], lay_clas, 'relu')
                model.compile(loss=ut.Wmulti_taskPFD(ww), optimizer="adam", metrics=['accuracy'])
                model.fit(X_trainr, y_MT, epochs=epo, batch_size=bs, verbose = 0, validation_split = vs, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = pat )])
                p = model.predict(X_testr)
                err1PFD = (1-accuracy_score(y_test, np.round(p[:,0])))
                mae1PFD = (mean_absolute_error(pri_test, p[:,1]))

                #----------------------------------------------------------
                #TPD M1
                #----------------------------------------------------------

                model = mo.mt1(X_trainr.shape[1], lay_clas, 'relu')
                model.compile(loss=ut.Wmulti_taskTPD(ww), optimizer="adam", metrics=['accuracy'])
                model.fit(X_trainr, y_MT, epochs=epo, batch_size=bs, verbose = 0, validation_split = vs, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = pat )])
                p = model.predict(X_testr)
                err1TPD = (1-accuracy_score(y_test, np.round(p[:,0])))
                mae1TPD = (mean_absolute_error(pri_test, p[:,1]))
                '''


                #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
                #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


                #MODEL2: Multi-task concatenation. Sigma and Temp independent of the instance.

                #----------------------------------------------------------
                #NORMAL M2
                #----------------------------------------------------------

                
                model  = mo.mt2(X_trainr.shape[1], lay_clas, lay_reg, 'relu')
                model.compile(loss=ut.WPmulti_task(ww), optimizer="adam", metrics=['accuracy'])
                model.fit(X_trainr, y_MT, epochs=epo, batch_size=bs, verbose = 0, validation_split = vs, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = pat)])
                p = model.predict(X_testr)
                err2 = (1-accuracy_score(y_test, np.round(p[:,0])))
                mae2 = (mean_absolute_error(pri_test, p[:,1]))

                #----------------------------------------------------------
                #PFD M2
                #----------------------------------------------------------
                model  = mo.mt2(X_trainr.shape[1], lay_clas, lay_reg, 'relu')
                model.compile(loss=ut.WPmulti_taskPFD(ww), optimizer="adam", metrics=['accuracy'])
                model.fit(X_trainr, y_MT, epochs=epo, batch_size=bs, verbose = 0, validation_split = vs, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = pat)])
                p = model.predict(X_testr)
                err2PFD = (1-accuracy_score(y_test, np.round(p[:,0])))
                mae2PFD = (mean_absolute_error(pri_test, p[:,1]))

                #----------------------------------------------------------
                #TPD M2
                #----------------------------------------------------------
                

                model  = mo.mt2(X_trainr.shape[1], lay_clas, lay_reg, 'relu')
                model.compile(loss=ut.WPmulti_taskTPD(ww), optimizer="adam", metrics=['accuracy'])
                model.fit(X_trainr, y_MT, epochs=epo, batch_size=bs, verbose = 0, validation_split = vs, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = pat)])
                p = model.predict(X_testr)
                err2TPD = (1-accuracy_score(y_test, np.round(p[:,0])))
                mae2TPD = (mean_absolute_error(pri_test, p[:,1]))
                print('lambda:', ww)
                print('sigma:', np.sqrt(np.exp(p[:,2][0])))
                print('temperature:', np.sqrt(np.exp(p[:,3][0])))
                print('lambda/2*sigma^2',  ww/(2*(np.exp(p[:,2][0]))))

  



                tf.keras.backend.clear_session()



                #Para el M2TPD calculamos los aciertos y fallos entre P, T y RL

                cm = []
                for i in range(len(y_pre_up)):
                    if np.round(p[:,0])[i] == y_pre_up[i] == y_test[i]:
                        cm.append('P = T = R')
                    if np.round(p[:,0])[i] == y_pre_up[i] != y_test[i]:
                        cm.append('P = T != R')
                    if np.round(p[:,0])[i] ==  y_test[i] != y_pre_up[i]:
                        cm.append('P = R != T')
                    if np.round(p[:,0])[i] != y_pre_up[i] == y_test[i]:
                        cm.append('P != R = T')


                col, xxx = np.unique(cm, return_counts = True)
                dict_cm_dc = dict(zip(col, xxx))



                if dict_cm_dc.get('P = T = R') is not None:
                    green = ((dict_cm_dc['P = T = R']/X_testr.shape[0])*100)
                else:
                    green = 0

                if dict_cm_dc.get('P = T != R') is not None:
                    yellow = ((dict_cm_dc['P = T != R']/X_testr.shape[0])*100)
                else:
                    yellow =  0

                if dict_cm_dc.get('P = R != T') is not None:
                    purple = ((dict_cm_dc['P = R != T']/X_testr.shape[0])*100)
                else:
                    purple = 0

                if dict_cm_dc.get('P != R = T') is not None:
                    red = ((dict_cm_dc['P != R = T']/X_testr.shape[0])*100)
                else:
                    red = 0


                #############
                #COMPARACIÓN CON EL MODELO BASE

                cmb = []
                for i in range(len(y_pre_up)):
                    if np.round(p[:,0])[i] == y_pre_b[i] == y_test[i]:
                        cmb.append('P = B = R')
                    if np.round(p[:,0])[i] == y_pre_b[i] != y_test[i]:
                        cmb.append('P = B != R')
                    if np.round(p[:,0])[i] ==  y_test[i] != y_pre_b[i]:
                        cmb.append('P = R != B')
                    if np.round(p[:,0])[i] != y_pre_b[i] == y_test[i]:
                        cmb.append('P != R = B')

                col, xxx = np.unique(cmb, return_counts = True)
                dict_cm_dcB = dict(zip(col, xxx))



                if dict_cm_dcB.get('P = B = R') is not None:
                    greenb = ((dict_cm_dcB['P = B = R']/X_testr.shape[0])*100)
                else:
                    greenb = 0

                if dict_cm_dcB.get('P = B != R') is not None:
                    redb = ((dict_cm_dcB['P = B != R']/X_testr.shape[0])*100)
                else:
                    redb = 0

                if dict_cm_dcB.get('P = R != B') is not None:
                    purpleb = ((dict_cm_dcB['P = R != B']/X_testr.shape[0])*100)
                else:
                    purpleb = 0

                if dict_cm_dcB.get('P != R = B') is not None:
                    yellowb = ((dict_cm_dcB['P != R = B']/X_testr.shape[0])*100)
                else:
                    yellowb = 0

                off_w = {'omega': ww,
                        #'error1': err1,
                        #'error1PFD': err1PFD,
                        #'error1TPD': err1TPD,
                        'error2': err2,
                        'error2PFD': err2PFD,
                        'error2TPD': err2TPD,
                        #'mae1': mae1,
                        #'mae1PFD': mae1PFD,
                        #'mae1TPD': mae1TPD,
                        'mae2': mae2,
                        'mae2PFD': mae2PFD,
                        'mae2TPD': mae2TPD,
                        'green': green,
                        'red': red,
                        'yellow': yellow,
                        'purple': purple,
                        'greenb': greenb,
                        'redb': redb,
                        'yellowb': yellowb,
                        'purpleb': purpleb
                        }   
                
                dw= pd.DataFrame(off_w, index = [0])
                dfw  = pd.concat([dfw, dw]).reset_index(drop = True)
                
            #dfw.groupby('omega').mean()


                
    off = {'name': t,
            'err_up':np.round(np.mean(err_up), 4),
            'err_b':  np.round(np.mean(err_b), 4)
            #'err_nn':  np.round(np.mean(err_nn), 4),
            #'err_nnPFD':  np.round(np.mean(err_nnPFD), 4),
            #'err_nnTPD':  np.round(np.mean(err_nnTPD), 4),
            #'mae_nn': np.round(np.mean(mae_nn), 4),
            #'GD': np.round(np.mean(err_gd), 4),
            #'PFD': np.round(np.mean(err_pfd), 4),
            #'TPD': np.round(np.mean(err_tpd), 4),
            #'LUPI_nn %': ut.LUPI_gain(np.round(np.mean(err_up), 4),  np.round(np.mean(err_b), 4), np.round(np.mean(err_nn), 4)),
            #'LUPI_nnPFD %': ut.LUPI_gain(np.round(np.mean(err_up), 4),  np.round(np.mean(err_b), 4), np.round(np.mean(err_nnPFD), 4)),
            #'LUPI_nnTPD%': ut.LUPI_gain(np.round(np.mean(err_up), 4),  np.round(np.mean(err_b), 4), np.round(np.mean(err_nnTPD), 4)),
            #'LUPI_gd %': ut.LUPI_gain(np.round(np.mean(err_up), 4),  np.round(np.mean(err_b), 4), np.round(np.mean(err_gd), 4)),
            #'LUPI_pfd %': ut.LUPI_gain(np.round(np.mean(err_up), 4),  np.round(np.mean(err_b), 4), np.round(np.mean(err_pfd), 4)),
            #'LUPI_tpd %': ut.LUPI_gain(np.round(np.mean(err_up), 4),  np.round(np.mean(err_b), 4), np.round(np.mean(err_tpd), 4)),
            }   
        
    df1 = pd.DataFrame(off, index = [0])
    row_to_repeat = df1.iloc[0]
    df1_omega = pd.DataFrame([row_to_repeat] * len(omega), columns=df1.columns).reset_index(drop = True)


    dffw = dfw.groupby('omega').mean().reset_index(drop = True)
    dffw['omega'] = omega

    df_concat = pd.concat([df1_omega, dffw], axis = 1)

    #df_concat['LUPI1'] = ut.LUPI_gain(df_concat['err_up'],  df_concat['err_b'], df_concat['error1'])
    #df_concat['LUPI1PFD'] = ut.LUPI_gain(df_concat['err_up'],  df_concat['err_b'], df_concat['error1PFD'])
    #df_concat['LUPI1TPD'] = ut.LUPI_gain(df_concat['err_up'],  df_concat['err_b'], df_concat['error1TPD'])

    df_concat['LUPI2'] = ut.LUPI_gain(df_concat['err_up'],  df_concat['err_b'], df_concat['error2'])
    df_concat['LUPI2PFD'] = ut.LUPI_gain(df_concat['err_up'],  df_concat['err_b'], df_concat['error2PFD'])
    df_concat['LUPI2TPD'] = ut.LUPI_gain(df_concat['err_up'],  df_concat['err_b'], df_concat['error2TPD'])


    dff  = pd.concat([dff, df_concat]).reset_index(drop = True)
            

    #SC (Sensitivity to class): % de predicciones de la clase mayoritaria
    #DC (Distance to change): Del rango total de la variable privilegiada, que % recorre hasta cambiar la clase.
    #SP (Sensitivity to probabilities): Desviación estándar de las probabilidades predichas por todo el rango de probabilidades de la variable privilegiada

    
v = '_'.join(args.dataset)
str_reg = [str(i) for i in args.l_reg]
lr = '-'.join(str_reg)
str_clas = [str(i) for i in args.l_clas]
lc = '-'.join(str_clas)

dff.to_csv('W_lambda_balance_only_REG_MT_' + v + '_' + lr + '_' + lc +  '_' + str(omega) + '_' + str(epo) + '_' + str(bs) + '_' + str(vs) + '_' + str(pat)+ '_' + str(n_iter)+ '.csv')
    
# %%
