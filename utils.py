from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, KFold
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Dropout, Activation, Input, Concatenate, Lambda
from keras import backend as K
import math
import matplotlib.pyplot as plt



#TODO: rename the parameters to be more informative
def skfold(X, y, n, r = 0, name = 'stratified'):
    """Returns the partitions for a K-fold cross validation

    Args:
        X (_type_): Input features
        y (_type_): Output feature
        n (_type_): number of splits
        r (int, optional): random seed. Defaults to 0.
        name (str, optional): _description_. Defaults to 'stratified'.

    Returns:
        _type_: _description_
    """
    if name == 'stratified':
        skf = StratifiedKFold(n_splits = n, shuffle = True, random_state = r)
    if name == 'time':
        skf = TimeSeriesSplit(n_splits = n)
    if name == 'kfold':
        skf = KFold(n_splits = n,  shuffle = True, random_state = r)
    
    d= {}
    j = 0
    for train_index, test_index in skf.split(X, y): 
            d['X_train' + str(j)] = X.loc[train_index]
            d['X_test' + str(j)] = X.loc[test_index]
            d['y_train' + str(j)] = y.loc[train_index]
            d['y_test' + str(j)] = y.loc[test_index]

            d['X_train' + str(j)].reset_index(drop = True, inplace = True)
            d['X_test' + str(j)].reset_index(drop = True, inplace = True)
            d['y_train' + str(j)].reset_index(drop = True, inplace = True)
            d['y_test' + str(j)].reset_index(drop = True, inplace = True)
            j+=1
    return d



def feat_correlation(X, y):
    """Selects as the privileged variable the most correlated one (w.r.t the class)

    Args:
        X (_type_): _description_
        y (_type_): _description_

    Returns:
        _type_: _description_
    """
    df = pd.concat([X, y], axis = 1)
    val1 = sorted(list(np.abs(df.corr(method = 'spearman')).iloc[-1]), reverse = True)[1]
    indice1 = list(np.abs(df.corr(method = 'spearman')).iloc[-1]).index(val1)
    pi_features = [X.columns[indice1]][0]
    return pi_features

def multi_feat_correlation(X, y, n):
    
    df = pd.concat([X, y], axis = 1)
    cm = df.corr(method='spearman')

    col = cm.columns[-1]
    val = cm[col]

    cd = pd.DataFrame({'V': val.index,
                       'C': np.abs(val.values)}).sort_values(by='C', ascending=False)

    pi_features = list(cd['V'].iloc[1:n+1])
    return pi_features



#TODO: rename the parameters
def train_test_fold(dr, h):
    """Given a dictionary with k-fold partitions, selects the h-th one

    Args:
        dr (_type_): _description_
        h (_type_): _description_

    Returns:
        _type_: _description_
    """
    X_train = dr['X_train' + str(h)]
    y_train = dr['y_train' + str(h)]
    X_test = dr['X_test' + str(h)]
    y_test = dr['y_test' + str(h)]
    
    X_train = X_train.reset_index(drop = True)
    y_train = y_train.reset_index(drop = True)
    X_test = X_test.reset_index(drop = True)
    y_test = y_test.reset_index(drop = True)
    return X_train, y_train, X_test, y_test






def LUPI_gain(ub, lb, x):
    """Gain of the privileged model (x) with respect to the upper- and lower-bound models (ub and lb)

    Args:
        ub (_type_): _description_
        lb (_type_): _description_
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    return ((x - lb) / (ub - lb) )*100



@tf.autograph.experimental.do_not_convert
#Definir en otro espacio py para eliminar el warning
def my_loss(y_true,y_pred):
    """Negative Log Likelihood loss

    Args:
        y_true (_type_): _description_
        y_pred (_type_): _description_

    Returns:
        _type_: _description_
    """
    mu=tf.slice(y_pred,[0,0],[-1,1])
    sigma=tf.math.exp(tf.slice(y_pred,[0,1],[-1,1])) 
    a=1/(tf.sqrt(2.*math.pi)*sigma)
    b1=tf.square(mu-y_true)
    b2=2*tf.square(sigma)
    b=b1/b2
    loss = tf.reduce_sum(-tf.math.log(a)+b,axis=0)
    return loss



#Loss for Generalized distillation algorithm



def bce_t(y_true, y_pred):
    lin=tf.slice(y_pred,[0,0],[-1,1])
    temp= tf.slice(y_pred,[0,1],[-1,1])**2 + 1e-5
    #O relu tambien valdria. Solo numeros positivos.
    ft = lin/(temp)
    #tf.print(lin)
    y_pr = keras.activations.sigmoid(ft)
    #tf.print(lin/temp)

    #if tf.reduce_any(tf.math.is_nan(lin)):
        #tf.print(lin)

    #print('------SEP--------')
    #tf.print(y_pr)

    b = tf.keras.losses.BinaryCrossentropy()(y_true, y_pr + 1e-5)
    return b

def loss_GD(T, l):
    def loss(y_true, y_pred):
        y_tr = y_true[:, 0]
        y_prob = y_true[:, 1]
        #Aquí estaba el problema con los nan añadiendo estas sumas se corrige
        ft = (-tf.math.log(1/(y_prob+1e-6) - 1 + 1e-6)) / T
        y_pr = 1 / (1 + tf.exp(-ft))
        #tf.print(y_pr)
        d1 = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(y_tr, y_pred)
        d2 = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(y_pr, y_pred)
        #tf.print(d2)
        #No puedo Categorical por el single layer del output y_pred
        #return (1-l)*d1 + l*d2
        return tf.reduce_mean( (1-l)*d1 + l*d2)
    return loss

def loss_TPD(T, beta, l):
    def loss(y_true, y_pred):
        y_tr = y_true[:, 0]
        y_prob = y_true[:, 1]
        d = y_true[:, 2]
        
        ft = (-tf.math.log(1/(y_prob+1e-6) - 1 + 1e-6)) / T
        y_pr = 1 / (1 + tf.exp(-ft))

        #BCE instance by instance
        bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        bce_inst = bce(y_pred, y_pr )
        bce_r = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(y_tr, y_pred)
        return tf.reduce_mean((1-l)*(bce_r) + l*(tf.math.multiply(d,bce_inst) - beta * tf.math.multiply(1-d, bce_inst))) 
    return loss



                
def multi_task(y_true, y_pred):
    y_tr = tf.reshape(y_true[:, 0], [-1,1])
    pri = tf.reshape(y_true[:, 1], [-1,1])


    #print(y_pred)
    c_pre, pi_pre, sigma_c, sigma_T  = tf.unstack(y_pred, num=4, axis=1)
    c_pre = tf.reshape(c_pre, [-1, 1]) 
    pi_pre = tf.reshape(pi_pre, [-1, 1]) 
    #tf.print(c_pre)

    l1 = (1/(2*tf.math.exp(sigma_c)))*tf.keras.losses.mean_squared_error(pri, pi_pre) + tf.math.log(tf.sqrt(tf.math.exp(sigma_c)))
    l2 = (1/(tf.math.exp(sigma_T)))*tf.keras.losses.binary_crossentropy(y_tr, c_pre) + tf.math.log(tf.sqrt(tf.math.exp(sigma_T)))
    return tf.reduce_mean(l1 + l2)





def multi_task_G(n):
    def loss(y_true, y_pred):
        y_true_clas = tf.reshape(y_true[:, 0], [-1, 1])
        y_true_regs = [tf.reshape(y_true[:, i+1], [-1, 1]) for i in range(n)]

        # Separar las salidas predichas
        SS = tf.unstack(y_pred, num=2+2*n, axis=1)
        c_pre = tf.reshape(SS[0], [-1,1]) 
        pri_out = [tf.reshape(SS[i+1], [-1]) for i in range(n)]
        sig_out = [tf.reshape(SS[i + n +1], [-1]) for i in range(n)]
        sig_temp = SS[2*n+1]

        # Pérdida de clasificación con temperatura
        l_classification = (1 / tf.math.exp(sig_temp)) * \
                        tf.keras.losses.binary_crossentropy(y_true_clas, c_pre) + \
                        tf.math.log(tf.math.sqrt(tf.math.exp(sig_temp)))

        # Pérdidas de regresión con sigma
        l_regression = []
        for i in range(n):
            l_reg = (1 / (2 * tf.math.exp(sig_out[i]))) * \
                    tf.keras.losses.mean_squared_error(y_true_regs[i], pri_out[i]) + \
                    tf.math.log(tf.math.sqrt(tf.math.exp(sig_out[i])))
            l_regression.append(l_reg)

        # Sumar todas las pérdidas
        total_loss = l_classification + tf.reduce_mean(l_regression, axis=0)
        return tf.reduce_mean(total_loss)

    return loss


def multi_task_GPFD(n):
    def loss(y_true, y_pred):
        y_true_clas = tf.reshape(y_true[:, 0], [-1, 1])
        y_true_regs = [tf.reshape(y_true[:, i+1], [-1, 1]) for i in range(n)]
        #y_upper = tf.reshape(y_true[:, -2], [-1,1])
        y_upper = tf.reshape(y_true[:, n+1], [-1,1])



        # Separar las salidas predichas
        SS = tf.unstack(y_pred, num=2+2*n, axis=1)
        c_pre = tf.reshape(SS[0], [-1,1]) 
        pri_out = [tf.reshape(SS[i+1], [-1]) for i in range(n)]
        sig_out = [tf.reshape(SS[i + n +1], [-1]) for i in range(n)]
        sig_temp = SS[2*n+1]

        # Pérdida de clasificación con temperatura
        l_classification = (1 / tf.math.exp(sig_temp)) * \
                        0.5*tf.keras.losses.binary_crossentropy(y_upper, c_pre) + \
                        0.5*tf.keras.losses.binary_crossentropy(y_true_clas, c_pre)+ \
                        tf.math.log(tf.math.sqrt(tf.math.exp(sig_temp)))

        # Pérdidas de regresión con sigma
        l_regression = []
        for i in range(n):
            l_reg = (1 / (2 * tf.math.exp(sig_out[i]))) * \
                    tf.keras.losses.mean_squared_error(y_true_regs[i], pri_out[i]) + \
                    tf.math.log(tf.math.sqrt(tf.math.exp(sig_out[i])))
            l_regression.append(l_reg)

        # Sumar todas las pérdidas
        total_loss = l_classification + tf.reduce_mean(l_regression, axis=0)
        return tf.reduce_mean(total_loss)

    return loss

def multi_task_GTPD(n):
    def loss(y_true, y_pred):
        y_true_clas = tf.reshape(y_true[:, 0], [-1, 1])
        y_true_regs = [tf.reshape(y_true[:, i+1], [-1, 1]) for i in range(n)]
        y_upper = tf.reshape(y_true[:, n+1], [-1,1])
        d =  y_true[:,  n+2]
        #y_upper = tf.reshape(y_true[:, -2], [-1,1])
        #d =  y_true[:,  -1]



        # Separar las salidas predichas
        SS = tf.unstack(y_pred, num=2+2*n, axis=1)
        c_pre = tf.reshape(SS[0], [-1,1]) 
        pri_out = [tf.reshape(SS[i+1], [-1]) for i in range(n)]
        sig_out = [tf.reshape(SS[i + n +1], [-1]) for i in range(n)]
        sig_temp = SS[2*n+1]

        # Pérdida de clasificación con temperatura
        bce_r = tf.keras.losses.binary_crossentropy(y_true_clas, c_pre)
        bce_inst = tf.keras.losses.binary_crossentropy(c_pre, y_upper)
        l_classification = (1 / tf.math.exp(sig_temp)) * \
                        (0.5*(bce_r) + 0.5*(tf.math.multiply(d,bce_inst) - tf.math.multiply(1-d, bce_inst))) +\
                        tf.math.log(tf.math.sqrt(tf.math.exp(sig_temp)))

        # Pérdidas de regresión con sigma
        l_regression = []
        for i in range(n):
            l_reg = (1 / (2 * tf.math.exp(sig_out[i]))) * \
                    tf.keras.losses.mean_squared_error(y_true_regs[i], pri_out[i]) + \
                    tf.math.log(tf.math.sqrt(tf.math.exp(sig_out[i])))
            l_regression.append(l_reg)

        # Sumar todas las pérdidas
        total_loss = l_classification + tf.reduce_mean(l_regression, axis=0)
        return tf.reduce_mean(total_loss)

    return loss




def multi_taskPFD(y_true, y_pred):
    y_tr = tf.reshape(y_true[:, 0], [-1,1])
    pri = tf.reshape(y_true[:, 1], [-1,1])
    y_upper = tf.reshape(y_true[:, 2], [-1,1])


    #print(y_pred)
    c_pre, pi_pre, sigma_c, sigma_T  = tf.unstack(y_pred, num=4, axis=1)
    c_pre = tf.reshape(c_pre, [-1, 1]) 
    pi_pre = tf.reshape(pi_pre, [-1, 1]) 
    #tf.print(c_pre)

    l1 = (1/(2*tf.math.exp(sigma_c)))*tf.keras.losses.mean_squared_error(pri, pi_pre) + tf.math.log(tf.sqrt(tf.math.exp(sigma_c)))
    l2 = (1/(tf.math.exp(sigma_T)))*(0.5*tf.keras.losses.binary_crossentropy(y_upper, c_pre) + 0.5*tf.keras.losses.binary_crossentropy(y_tr, c_pre) ) + tf.math.log(tf.sqrt(tf.math.exp(sigma_T)))
    return tf.reduce_mean(l1 + l2)



def multi_taskTPD(y_true, y_pred):
    y_tr = tf.reshape(y_true[:, 0], [-1,1])
    pri = tf.reshape(y_true[:, 1], [-1,1])
    y_upper = tf.reshape(y_true[:, 2], [-1,1])
    d = y_true[:, 3]

    c_pre, pi_pre, sigma_c, sigma_T  = tf.unstack(y_pred, num=4, axis=1)
    c_pre = tf.reshape(c_pre, [-1, 1]) 
    pi_pre = tf.reshape(pi_pre, [-1, 1]) 

    l1 = (1/(2*tf.math.exp(sigma_c)))*tf.keras.losses.mean_squared_error(pri, pi_pre) + tf.math.log(tf.sqrt(tf.math.exp(sigma_c)))
    bce_r = tf.keras.losses.binary_crossentropy(y_tr, c_pre)
    bce_inst = tf.keras.losses.binary_crossentropy(c_pre, y_upper) 
    l2 = (1/(tf.math.exp(sigma_T)))*(0.5*(bce_r) + 0.5*(tf.math.multiply(d,bce_inst) - tf.math.multiply(1-d, bce_inst))) + tf.math.log(tf.sqrt(tf.math.exp(sigma_T)))
    return tf.reduce_mean(l1 + l2)




def dtc(pri_test, X_train, X_testr, model, pi_features, step = 0.01): 
    ranges = [np.arange(min(pri_test), max(pri_test), step) for i in pri_test]
    X_testr.insert(X_train.columns.get_loc(pi_features), pi_features, ranges)
    ex = X_testr.explode(pi_features).reset_index(drop = True)
    ex[pi_features] = ex[pi_features].astype('float64')
    pre_soft = np.ravel(model.predict(ex))
    df_ex = pd.DataFrame({'pi': ex[pi_features], 'pre': np.round(pre_soft)})


    dc = []
    no_dc = []
    c = 0
    for j in range(0, len(pre_soft), len(ranges[0])):
        subset = df_ex[j:j+len(ranges[0])]
        ind =   min(range(len(subset['pi'])), key=lambda i: abs(subset['pi'].iloc[i] - pri_test[c]))
        num = subset['pi'].iloc[ind]

        # Buscar el primer cambio de clase hacia la derecha
        cd = next((i for i in range(ind + 1, len(subset)) if subset['pre'].iloc[i] != subset['pre'].iloc[ind]), None)
        ci = next((i for i in range(ind - 1, -1, -1) if subset['pre'].iloc[i] != subset['pre'].iloc[ind]), None)

        #ci = next((i for i in range(ind - 1, -1, -1) if subset[i] == 1), None)

        if cd != None:
            dc_der = subset['pi'].iloc[cd] - pri_test[c]

        if ci != None:
            dc_iz = subset['pi'].iloc[ci] - pri_test[c]

    
        if (cd != None) & (ci != None):
            if np.abs(dc_der) > np.abs(dc_iz):
                dc.append(dc_iz)
            else:
                dc.append(dc_der)
        if (cd == None) & (ci != None):
            dc.append(dc_iz)

        if (cd != None) & (ci == None):
            dc.append(dc_der)

        if (cd == None) & (ci == None):
            no_dc.append(c)
            dc.append(None)

        c+=1
    return dc, no_dc







#------------------------------------------------------------------
#Solo la loss penalizacion
#loss_logs = {'l1': [], 'l2': [], 'total_loss': []}

def WPmulti_task(w):
    def loss(y_true, y_pred):
        y_tr = tf.reshape(y_true[:, 0], [-1,1])
        pri = tf.reshape(y_true[:, 1], [-1,1])

        #print(y_pred)
        c_pre, pi_pre, sigma_c, sigma_T  = tf.unstack(y_pred, num=4, axis=1)
        c_pre = tf.reshape(c_pre, [-1, 1]) 
        pi_pre = tf.reshape(pi_pre, [-1, 1]) 


        l1w =  w * ((1/(2*tf.math.exp(sigma_c)))*tf.keras.losses.mean_squared_error(pri, pi_pre)) + tf.math.log(tf.sqrt(tf.math.exp(sigma_c)))
        l2 = (1/(tf.math.exp(sigma_T)))*tf.keras.losses.binary_crossentropy(y_tr, c_pre) + tf.math.log(tf.sqrt(tf.math.exp(sigma_T)))

        total_loss = tf.reduce_mean(l1w + l2)

        return total_loss
    return loss

def WPmulti_taskPFD(w, loss_logs):
    def loss(y_true, y_pred):
        y_tr = tf.reshape(y_true[:, 0], [-1,1])
        pri = tf.reshape(y_true[:, 1], [-1,1])
        y_upper = tf.reshape(y_true[:, 2], [-1,1])


        #print(y_pred)
        c_pre, pi_pre, sigma_c, sigma_T  = tf.unstack(y_pred, num=4, axis=1)
        c_pre = tf.reshape(c_pre, [-1, 1]) 
        pi_pre = tf.reshape(pi_pre, [-1, 1]) 
        #tf.print(c_pre)

        l1w =  w * ((1/(2*tf.math.exp(sigma_c)))*tf.keras.losses.mean_squared_error(pri, pi_pre)) + tf.math.log(tf.sqrt(tf.math.exp(sigma_c)))
        l2 = (1/(tf.math.exp(sigma_T)))*(0.5*tf.keras.losses.binary_crossentropy(y_upper, c_pre) + 0.5*tf.keras.losses.binary_crossentropy(y_tr, c_pre) ) + tf.math.log(tf.sqrt(tf.math.exp(sigma_T)))

        total_loss = tf.reduce_mean(l1w + l2)
        return total_loss
    return loss

def WPmulti_taskTPD(w):
    def loss(y_true, y_pred):
        y_tr = tf.reshape(y_true[:, 0], [-1,1])
        pri = tf.reshape(y_true[:, 1], [-1,1])
        y_upper = tf.reshape(y_true[:, 2], [-1,1])
        #d =  tf.reshape(y_true[:, 3], [-1,1])
        d = y_true[:, 3]

        #print(y_pred)
        c_pre, pi_pre, sigma_c, sigma_T  = tf.unstack(y_pred, num=4, axis=1)
        c_pre = tf.reshape(c_pre, [-1, 1]) 
        pi_pre = tf.reshape(pi_pre, [-1, 1]) 
        #tf.print(c_pre)

        l1w =  w * ((1/(2*tf.math.exp(sigma_c)))*tf.keras.losses.mean_squared_error(pri, pi_pre)) + tf.math.log(tf.sqrt(tf.math.exp(sigma_c)))
        bce_r = tf.keras.losses.binary_crossentropy(y_tr, c_pre)
        bce_inst = tf.keras.losses.binary_crossentropy(c_pre, y_upper) 
        l2 = (1/(tf.math.exp(sigma_T)))*(0.5*(bce_r) + 0.5*(tf.math.multiply(d,bce_inst) - tf.math.multiply(1-d, bce_inst))) + tf.math.log(tf.sqrt(tf.math.exp(sigma_T)))
  
        total_loss = tf.reduce_mean(l1w + l2)
        
        return total_loss
    return loss





