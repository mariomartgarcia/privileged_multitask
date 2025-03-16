# MORE ABOUT THIS FILE:

# Previously called GNN_MT_ALL_inter.py. This file has been used to generate the scatter plots 
# of the paper.


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


#python NN_MT_ALL_inter.py -dataset obesity  -epo 2 -bs 128 -vs 0.2 -pat 1 -iter 1
#python GNN_MT_ALL_inter.py -dataset phishing obesity diabetes wm phoneme magic_telescope mozilla elect wine kin elevators abalone wind sil -l_reg 20 20 -l_clas 20 20 -epo 1000 -bs 128 -vs 0.2 -pat 5 -iter 1
#python NN_MT_ALL_DC.py -dataset elect wine kin elevators -l_reg 20 20 -l_clas 20 20 -epo 1000 -bs 128 -vs 0.2 -pat 5 -iter 1
#python NN_MT_ALL_DC.py -dataset abalone wind sil -l_reg 20 20 -l_clas 20 20 -epo 1000 -bs 128 -vs 0.2 -pat 5 -iter 1

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
text = ['diabetes']
epo = 100
bs = 128
vs = 0.2
pat = 5
val_drp = 0
n_iter = 1
ran = np.random.randint(1000, size = n_iter)

lay_reg = [20,20]
lay_clas = [20,20]
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

    mae1, mae1PFD, mae1TPD  = [[] for i in range(3)]
    mae2, mae2PFD, mae2TPD = [[] for i in range(3)]

    err_nn, err_nnPFD, err_nnTPD, mae_nn = [[] for i in range(4)]

    per, std_per, dist = [[] for i in range(3)]

    green, red, purple, yellow, dc_per = [[] for i in range(5)]
    greenb, redb, purpleb, yellowb = [[] for i in range(4)]

    dc, err, com, comb, comt = [[] for i in range(5)]

    #For each fold (k is the random seed of each fold)
    for k in ran:
        #Create a dictionary with all the fold partitions
        dr = ut.skfold(X, pd.Series(list(y)), cv, r = k, name = 'stratified')
        #Process each fold individually
        point = 1
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


            dc_raw, no_dc = ut.dtc(pri_test, X_train, X_testr, model, pi_features, 0.01)
            dcc = [dc_raw[i] for i in range(len(dc_raw)) if i not in no_dc]

            dc.extend(dcc)
            X_testr = X_testr.drop(pi_features, axis = 1)
                

            #----------------------------------------------------------
            #LOWER
            #----------------------------------------------------------

            model =  mo.nn_binary_clasification( X_trainr.shape[1], lay_clas, 'relu')
            model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
            model.fit(X_trainr, y_train, epochs=epo, batch_size=bs, verbose = 0, validation_split = vs, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = pat)])

            #Measure test error
            y_pre_b = np.ravel([np.round(i) for i in model.predict(X_testr)])

            err_b.append(1-accuracy_score(y_test, y_pre_b))

    



            y_MT = np.column_stack([np.ravel(y_train), np.ravel(pri), np.ravel(y_pre_prob), np.ravel(delta_i)])


            #----------------------------------------------------------
            #TPD M2
            #----------------------------------------------------------

            model  = mo.mt2(X_trainr.shape[1], lay_clas, lay_reg, 'relu')
            model.compile(loss=ut.multi_taskTPD, optimizer="adam", metrics=['accuracy'])
            model.fit(X_trainr, y_MT, epochs=epo, batch_size=bs, verbose = 0, validation_split = vs, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = pat)])
            p = model.predict(X_testr)
            err2TPD.append(1-accuracy_score(y_test, np.round(p[:,0])))
            mae2TPD.append(mean_absolute_error(pri_test, p[:,1]))

            def error(pri_test, r):
                pr = np.ravel(pri_test)
                pp = np.ravel(r)
                err = [ pp[i] - pr[i] for i in range(len(pr))]
                return err
            
            #Todos los errores, me quedo con los que tienen dc
            err_raw = error(pri_test, p[:,1])
            err_dc = [err_raw[i] for i in range(len(err_raw)) if i not in no_dc]
            err.extend(err_dc)


            #Instancias con mismo output para el MT2 y Teacher, me quedo con las que tienen dc  
            #suc_raw = [1 if y_pre_up[i] ==  np.round(p[:,0])[i] else 0 for i in range(len(y_pre_up))]
            #suc_dc = [suc_raw[i] for i in range(len(suc_raw)) if i not in no_dc]
            #Instancias con mismo output para el MT2 y Real Labels, me quedo con las que tienen dc  
            #suc_raw_t = [1 if y_test[i] ==  np.round(p[:,0])[i] else 0 for i in range(len(y_test))]
            #suct_dc = [suc_raw_t[i] for i in range(len(suc_raw_t)) if i not in no_dc]

            #Aciertos y fallos entre P, T y RL
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



            cm_dc = [cm[i] for i in range(len(cm)) if i not in no_dc]
            com.extend([cm[i] for i in range(len(cm)) if i not in no_dc])

            col, xxx = np.unique(cm_dc, return_counts = True)
            dict_cm_dc = dict(zip(col, xxx))



            if dict_cm_dc.get('P = T = R') is not None:
                green.append((dict_cm_dc['P = T = R']/(X_testr.shape[0] - len(no_dc)))*100)
            else:
                green.append(0)

            if dict_cm_dc.get('P = T != R') is not None:
                yellow.append((dict_cm_dc['P = T != R']/(X_testr.shape[0] - len(no_dc)))*100)
            else:
                yellow.append(0)

            if dict_cm_dc.get('P = R != T') is not None:
                purple.append((dict_cm_dc['P = R != T']/(X_testr.shape[0] - len(no_dc)))*100)
            else:
                purple.append(0)

            if dict_cm_dc.get('P != R = T') is not None:
                red.append((dict_cm_dc['P != R = T']/(X_testr.shape[0] - len(no_dc)))*100)
            else:
                red.append(0)


            #yellow.append((dict_cm_dc['P = T != R']/(X_testr.shape[0] - len(no_dc)))*100)
            #purple.append((dict_cm_dc['P = R != T']/(X_testr.shape[0] - len(no_dc)))*100)
            #red.append((dict_cm_dc['P != R = T']/(X_testr.shape[0] - len(no_dc)))*100)


            print(t)
            print(X_testr.shape[0])
            print(len(no_dc))

            dc_per.append(((X_testr.shape[0] - len(no_dc))/X_testr.shape[0])*100)


            #############
            #COMPARACIÓN CON EL MODELO BASE Y LOS LABELS REALES

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



            cmb_dc = [cmb[i] for i in range(len(cmb)) if i not in no_dc]
            comb.extend(cmb_dc)

            col, xxx = np.unique(cmb_dc, return_counts = True)
            dict_cm_dcB = dict(zip(col, xxx))



            if dict_cm_dcB.get('P = B = R') is not None:
                greenb.append((dict_cm_dcB['P = B = R']/(X_testr.shape[0] - len(no_dc)))*100)
            else:
                greenb.append(0)

            if dict_cm_dcB.get('P = B != R') is not None:
                redb.append((dict_cm_dcB['P = B != R']/(X_testr.shape[0] - len(no_dc)))*100)
            else:
                redb.append(0)

            if dict_cm_dcB.get('P = R != B') is not None:
                purpleb.append((dict_cm_dcB['P = R != B']/(X_testr.shape[0] - len(no_dc)))*100)
            else:
                purpleb.append(0)

            if dict_cm_dcB.get('P != R = B') is not None:
                yellowb.append((dict_cm_dcB['P != R = B']/(X_testr.shape[0] - len(no_dc)))*100)
            else:
                yellowb.append(0)

            #############
            #COMPARACIÓN CON EL MODELO TEACHER
            
            cmt = []
            for i in range(len(y_pre_up)):
                if np.round(p[:,0])[i] == y_pre_b[i] == y_pre_up[i]:
                    cmt.append('P = B = T')
                if np.round(p[:,0])[i] == y_pre_b[i] != y_pre_up[i]:
                    cmt.append('P = B != T')
                if np.round(p[:,0])[i] ==  y_pre_up[i] != y_pre_b[i]:
                    cmt.append('P = T != B')
                if np.round(p[:,0])[i] != y_pre_b[i] == y_pre_up[i]:
                    cmt.append('P != T = B')



            cmt_dc = [cmt[i] for i in range(len(cmt)) if i not in no_dc]
            comt.extend(cmt_dc)



            
            #GRÁFICA
            
            import seaborn as sns
            import matplotlib.pyplot as plt


            df_scat = pd.DataFrame({'dc': dc, 'err': err, 'com':com, 'comb':comb, 'comt':comt})

            dc_min, dc_max = df_scat['dc'].min(), df_scat['dc'].max()
            err_min, err_max = df_scat['err'].min(), df_scat['err'].max()
            # Determinar el rango común para ajustar la línea y = x
            line_min = max(dc_min, err_min)  # Máximo entre los mínimos
            line_max = min(dc_max, err_max)  # Mínimo entre los máximos
            
            sns.set_style("whitegrid")
            scatter = sns.jointplot(
                data=df_scat,
                x="dc",  # Eje x
                y="err",  # Eje y
                hue="com",  # Columna categórica para colores
                kind="scatter",
                palette={'P != R = T': 'red', 'P = T = R': 'green', 'P = T != R': 'orange',  'P = R != T': 'purple'},  
                #palette={'P != R = B': 'orange', 'P = B = R': 'green', 'P = B != R': 'red',  'P = R != B': 'purple'}, 
                s=100,  # Tamaño de los puntos
                height=8, # Tamaño de la figura
            )

            # Personalizar el eje principal del jointplot
            scatter.ax_joint.axhline(0, color='black', linewidth=0.8)  # Línea horizontal
            scatter.ax_joint.axvline(0, color='black', linewidth=0.8)  # Línea vertical
            xx = np.linspace(line_min, line_max, 100)  # Ajustar la línea a los valores calculados
            scatter.ax_joint.plot(xx, xx, label='DR = DC', color='black', linestyle='-.')  # Línea diagonal
            #scatter.ax_joint.set_title(t.upper() + ' | M2TPD | UPPER LABELS', fontsize=16)  # Título
            scatter.ax_joint.set_ylabel('Distance to Real (DR)', fontsize=14, fontweight='bold')  # Etiqueta eje Y
            scatter.ax_joint.set_xlabel('Distance to Change (DC)', fontsize=14, fontweight='bold')  # Etiqueta eje X
            scatter.ax_joint.tick_params(axis='x', labelsize=12)  # Tamaño de las etiquetas eje X
            scatter.ax_joint.tick_params(axis='y', labelsize=12)  # Tamaño de las etiquetas eje Y

            # Ajustar leyenda
            scatter.ax_joint.legend(loc="upper left", fontsize=12)
            plt.savefig('image/dc_11_02/' + str(point) + '_' + 'REAL' + '_'  + t.lower() + '.pdf', format='pdf', transparent=True, dpi=300, bbox_inches='tight')
            point += 1
            '''
            sns.set_style("whitegrid")
            scatter = sns.jointplot(
                data=df_scat,
                x="dc",  # Eje x
                y="err",  # Eje y
                hue="com",  # Columna categórica para colores
                kind="scatter",
                palette={'P != R = T': 'red', 'P = T = R': 'green', 'P = T != R': 'orange',  'P = R != T': 'purple'},  
                #palette={'P != T = B': 'orange', 'P = B = T': 'green', 'P = B != T': 'red',  'P = T != B': 'purple'}, 
                s=100,  # Tamaño de los puntos
                height=8, # Tamaño de la figura
            )

            # Personalizar el eje principal del jointplot
            scatter.ax_joint.axhline(0, color='black', linewidth=0.8)  # Línea horizontal
            scatter.ax_joint.axvline(0, color='black', linewidth=0.8)  # Línea vertical
            xx = np.linspace(line_min, line_max, 100)  # Ajustar la línea a los valores calculados
            scatter.ax_joint.plot(xx, xx, label='y = x', color='black', linestyle='-.')  # Línea diagonal
            #scatter.ax_joint.set_title(t.upper() + ' | M2TPD | UPPER LABELS', fontsize=16)  # Título
            scatter.ax_joint.set_ylabel('ERR', fontsize=14)  # Etiqueta eje Y
            scatter.ax_joint.set_xlabel('DC', fontsize=14)  # Etiqueta eje X
            scatter.ax_joint.tick_params(axis='x', labelsize=12)  # Tamaño de las etiquetas eje X
            scatter.ax_joint.tick_params(axis='y', labelsize=12)  # Tamaño de las etiquetas eje Y

            # Ajustar leyenda
            scatter.ax_joint.legend(loc="upper left", fontsize=12, title="SUC")
            plt.savefig('image/dc_20/' + str(point) + '_'  + t.lower() + '.pdf', format='pdf', transparent=True, dpi=300, bbox_inches='tight')

            point += 1

            
            
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
            '''
            tf.keras.backend.clear_session()
    
    #Save the results
    off = {'name' : t,
           'err_up':np.round(np.mean(err_up), 4),
           'err_b':  np.round(np.mean(err_b), 4),
           'error2TPD': np.round(np.mean(err2TPD), 4),
           'LUPI2TPD %': ut.LUPI_gain(np.round(np.mean(err_up), 4),  np.round(np.mean(err_b), 4), np.round(np.mean(err2TPD), 4)),
           'mae2TPD': np.round(np.mean(mae2TPD), 4),
           'err':  np.round(np.mean(err), 4),
           'dc':  np.round(np.mean(dc), 4),
           'green':  np.round(np.mean(green), 4),
           'red':  np.round(np.mean(red), 4),
           'yellow':  np.round(np.mean(yellow), 4),
           'purple':  np.round(np.mean(purple), 4),
           'err_std':  np.round(np.std(err), 4),
           'dc_std':  np.round(np.std(dc), 4),
           'green_std':  np.round(np.std(green), 4),
           'red_std':  np.round(np.std(red), 4),
           'yellow_std':  np.round(np.std(yellow), 4),
           'purple_std':  np.round(np.std(purple), 4),
            'greenb':  np.round(np.mean(greenb), 4),
            'redb':  np.round(np.mean(redb), 4),
            'yellowb':  np.round(np.mean(yellowb), 4),
            'purpleb':  np.round(np.mean(purpleb), 4),
            'greenb_std':  np.round(np.std(greenb), 4),
            'redb_std':  np.round(np.std(redb), 4),
            'yellowb_std':  np.round(np.std(yellowb), 4),
            'purpleb_std':  np.round(np.std(purpleb), 4)
           }   
    

    #SC (Sensitivity to class): % de predicciones de la clase mayoritaria
    #DC (Distance to change): Del rango total de la variable privilegiada, que % recorre hasta cambiar la clase.
    #SP (Sensitivity to probabilities): Desviación estándar de las probabilidades predichas por todo el rango de probabilidades de la variable privilegiada

    
    df1 = pd.DataFrame(off, index = [0])
        
    dff  = pd.concat([dff, df1]).reset_index(drop = True)


v = '_'.join(args.dataset)
str_reg = [str(i) for i in args.l_reg]
lr = '-'.join(str_reg)
str_clas = [str(i) for i in args.l_clas]
lc = '-'.join(str_clas)

dff.to_csv('perct_MT_' + v + '_' + lr + '_' + lc + '_' + str(epo) + '_' + str(bs) + '_' + str(vs) + '_' + str(pat)+ '_' + str(n_iter)+ '.csv')
    
# %%
