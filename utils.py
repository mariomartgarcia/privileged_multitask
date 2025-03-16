import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

def LUPI_gain(ub, lb, x):
    return ((x - lb) / (ub - lb) )*100


def load_image_pairs_with_labels(data_dir, classes, input_size=(64, 64), priv = 'NIR'):
    rgb_paths = []
    nir_paths = []
    labels = []  # Guardar etiquetas: 0 para 'Highway', 1 para 'River'

    for label, class_name in enumerate(classes):
        rgb_folder = os.path.join(data_dir, f'{class_name}RGB')
        nir_folder = os.path.join(data_dir, f'{class_name}' + priv)

        for file_name in os.listdir(rgb_folder):
            base_name = "_".join(file_name.split('_')[:-1]) 

            rgb_path = os.path.join(rgb_folder, base_name + '_RGB.png')
            nir_path = os.path.join(nir_folder, base_name + '_' + priv + '.png')

            if os.path.exists(rgb_path) and os.path.exists(nir_path):
                rgb_paths.append(rgb_path)
                nir_paths.append(nir_path)
                labels.append(label)  # Etiqueta basada en el índice de `classes`
            else:
                print('Warning mismatch')
                print(rgb_path)
                print(nir_path)

    # Load and preprocess images
    def preprocess_image(image_path):
        img = load_img(image_path, target_size=input_size)
        return img_to_array(img) / 255.0  # Normalize to [0, 1]

    rgb_images = np.array([preprocess_image(path) for path in rgb_paths])
    nir_images = np.array([preprocess_image(path)[:, :, 0:1] for path in nir_paths])  # Grayscale NIR
    labels = np.array(labels)  # Convertir las etiquetas a numpy array
    
    return rgb_images, nir_images, labels




def load_image_pairs_with_labels_s2_s1(data_dir, classes, input_size=(256, 256)):
    s2_paths = []
    s1_paths = []
    labels = []  # Etiquetas: por ejemplo, 0 para la primera clase, 1 para la segunda

    for label, class_name in enumerate(classes):
        s2_folder = os.path.join(data_dir, f'{class_name}_s2')
        s1_folder = os.path.join(data_dir, f'{class_name}_s1')

        for file_name in os.listdir(s2_folder):

            parts = file_name.split('_')
            p1 = "_".join(parts[:2]) 
            p2 = "_".join(parts[3:]) 
            

            s2_path = os.path.join(s2_folder, p1 + '_s2_' +  p2)
            s1_path = os.path.join(s1_folder, p1 + '_s1_' +  p2)

            if os.path.exists(s2_path) and os.path.exists(s1_path):
                s2_paths.append(s2_path)
                s1_paths.append(s1_path)
                labels.append(label)  # Etiqueta basada en el índice de `classes`
            else:
                print('Warning mismatch')
                print(s2_path)
                print(s1_path)

    # Preprocesar imágenes
    def preprocess_image(image_path):
        img = load_img(image_path, target_size=input_size)
        return img_to_array(img) / 255.0  # Normalizar a [0, 1]

    # Cargar y procesar imágenes
    s2_images = np.array([preprocess_image(path) for path in s2_paths])  # Imágenes S2
    s1_images = np.array([preprocess_image(path)[:, :, 0:1] for path in s1_paths])  # Imágenes S1 en escala de grises
    labels = np.array(labels)  # Convertir etiquetas a array numpy
    
    return s2_images, s1_images, labels



def concatenate_images(rgb_images, nir_predictions):
    expand = np.repeat(nir_predictions, 3, axis=-1)
    pre_pri = np.concatenate([rgb_images, expand], axis=1)
    return pre_pri  

def expand_array(arr, dim = (1, 64, 64, 1)):
    expanded_array = np.expand_dims(arr, axis=(-1, -2, -3))  
    ex = np.tile(expanded_array, dim) 
    return ex

def loss_TPD(T, beta, l):
    def loss(y_true, y_pred):
        y_tr = y_true[:, 0]
        y_prob = y_true[:, 1]
        d = y_true[:, 2]
        
        ft = (-tf.math.log(1/(y_prob+1e-6) - 1 + 1e-6)) / T
        y_pr = 1 / (1 + tf.exp(-ft))

        bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        bce_inst = bce(y_pred, y_pr )
        bce_r = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(y_tr, y_pred)
        return tf.reduce_mean((1-l)*(bce_r) + l*(tf.math.multiply(d,bce_inst) - beta * tf.math.multiply(1-d, bce_inst))) 
    return loss

def loss_GD(T, l):
    def loss(y_true, y_pred):
        y_tr = y_true[:, 0]
        y_prob = y_true[:, 1]

        ft = (-tf.math.log(1/(y_prob+1e-6) - 1 + 1e-6)) / T
        y_pr = 1 / (1 + tf.exp(-ft))
        d1 = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(y_tr, y_pred)
        d2 = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(y_pr, y_pred)
        return tf.reduce_mean( (1-l)*d1 + l*d2)
    return loss

def loss_MT(y_true, y_pred):
    pri = y_true[:, :, :, 0]
    y_tr = tf.reshape(tf.reduce_max(y_true[:, :, :, 1], axis=(1, 2)), [-1, 1])

    pi_pre = y_pred[:, :, :, 0]
    c_pre = tf.reshape(tf.reduce_max(y_pred[:, :, :, 1], axis=(1, 2)), [-1, 1])
    sigma = tf.reduce_max(y_pred[:, :, :, 2], axis=(1, 2))
    temperature = tf.reduce_max(y_pred[:, :, :, 3], axis=(1, 2))

    l1 = (1/(2*tf.math.exp(sigma)))*tf.reduce_mean(tf.square(pi_pre - pri), axis=[1, 2]) + tf.math.log(tf.sqrt(tf.math.exp(sigma)))
    l2 = (1/(tf.math.exp(temperature)))*tf.keras.losses.binary_crossentropy(y_tr, c_pre)  + tf.math.log(tf.sqrt(tf.math.exp(temperature)))

    return tf.reduce_mean(l1 + l2)



def loss_MT_TPD(y_true, y_pred):
    pri = y_true[:, :, :, 0]
    y_tr = tf.reshape(tf.reduce_max(y_true[:, :, :, 1], axis=(1, 2)), [-1, 1])
    y_upper = tf.reshape(tf.reduce_max(y_true[:, :, :, 2], axis=(1, 2)), [-1, 1])
    #d = tf.reshape(tf.reduce_max(y_true[:, :, :, 3], axis=(1, 2)), [-1, 1])
    d = tf.reduce_max(y_true[:, :, :, 3], axis=(1, 2))


    pi_pre = y_pred[:, :, :, 0]
    c_pre = tf.reshape(tf.reduce_max(y_pred[:, :, :, 1], axis=(1, 2)), [-1, 1])
    #sigma = tf.reshape(tf.reduce_max(y_pred[:, :, :, 2], axis=(1, 2)), [-1, 1])
    #temperature = tf.reshape(tf.reduce_max(y_pred[:, :, :, 3], axis=(1, 2)), [-1, 1])
    sigma = tf.reduce_max(y_pred[:, :, :, 2], axis=(1, 2))
    temperature = tf.reduce_max(y_pred[:, :, :, 3], axis=(1, 2))

    l1 = (1/(2*tf.math.exp(sigma)))*tf.reduce_mean(tf.square(pi_pre - pri), axis=[1, 2]) + tf.math.log(tf.sqrt(tf.math.exp(sigma)))
    bce_r = tf.keras.losses.binary_crossentropy(y_tr, c_pre)
    bce_inst = tf.keras.losses.binary_crossentropy(c_pre, y_upper) 
    l2 = (1/(tf.math.exp(temperature)))*(0.5*(bce_r) + 0.5*(tf.math.multiply(d,bce_inst) - tf.math.multiply(1-d, bce_inst))) + tf.math.log(tf.sqrt(tf.math.exp(temperature)))
    return tf.reduce_mean(l1 + l2)


def loss_MT_PFD(y_true, y_pred):
    pri = y_true[:, :, :, 0]
    y_tr = tf.reshape(tf.reduce_max(y_true[:, :, :, 1], axis=(1, 2)), [-1, 1])
    y_upper = tf.reshape(tf.reduce_max(y_true[:, :, :, 2], axis=(1, 2)), [-1, 1])

    pi_pre = y_pred[:, :, :, 0]
    c_pre = tf.reshape(tf.reduce_max(y_pred[:, :, :, 1], axis=(1, 2)), [-1, 1])
    sigma = tf.reduce_max(y_pred[:, :, :, 2], axis=(1, 2))
    temperature = tf.reduce_max(y_pred[:, :, :, 3], axis=(1, 2))

    l1 = (1/(2*tf.math.exp(sigma)))*tf.reduce_mean(tf.square(pi_pre - pri), axis=[1, 2]) + tf.math.log(tf.sqrt(tf.math.exp(sigma)))
    bce = 0.5*tf.keras.losses.binary_crossentropy(y_upper, c_pre) + 0.5*tf.keras.losses.binary_crossentropy(y_tr, c_pre) 
    l2 = (1/(tf.math.exp(temperature)))*(bce) + tf.math.log(tf.sqrt(tf.math.exp(temperature)))
    return tf.reduce_mean(l1 + l2)


# PRUEBA DE LOSS EPSILON = 0. ¿Influye el coste computacional (tener una red más grande en el rendimiento final)?
# Dos conjuntos de losses se proponen: (A) Epsilon = 0 (B) Epsilon = 0.  No regularization.

#(A) Epsilon = 0
def loss_MT_A(y_true, y_pred):
    #pri = y_true[:, :, :, 0]
    y_tr = tf.reshape(tf.reduce_max(y_true[:, :, :, 1], axis=(1, 2)), [-1, 1])

    #pi_pre = y_pred[:, :, :, 0]
    c_pre = tf.reshape(tf.reduce_max(y_pred[:, :, :, 1], axis=(1, 2)), [-1, 1])
    #sigma = tf.reduce_max(y_pred[:, :, :, 2], axis=(1, 2))
    temperature = tf.reduce_max(y_pred[:, :, :, 3], axis=(1, 2))

    #l1 = (1/(2*tf.math.exp(sigma)))*tf.reduce_mean(tf.square(pi_pre - pri), axis=[1, 2]) + tf.math.log(tf.sqrt(tf.math.exp(sigma)))
    l2 = (1/(tf.math.exp(temperature)))*tf.keras.losses.binary_crossentropy(y_tr, c_pre)  + tf.math.log(tf.sqrt(tf.math.exp(temperature)))

    return tf.reduce_mean(l2)

def loss_MT_PFD_A(y_true, y_pred):
    #pri = y_true[:, :, :, 0]
    y_tr = tf.reshape(tf.reduce_max(y_true[:, :, :, 1], axis=(1, 2)), [-1, 1])
    y_upper = tf.reshape(tf.reduce_max(y_true[:, :, :, 2], axis=(1, 2)), [-1, 1])

    #pi_pre = y_pred[:, :, :, 0]
    c_pre = tf.reshape(tf.reduce_max(y_pred[:, :, :, 1], axis=(1, 2)), [-1, 1])
    #sigma = tf.reduce_max(y_pred[:, :, :, 2], axis=(1, 2))
    temperature = tf.reduce_max(y_pred[:, :, :, 3], axis=(1, 2))

    #l1 = (1/(2*tf.math.exp(sigma)))*tf.reduce_mean(tf.square(pi_pre - pri), axis=[1, 2]) + tf.math.log(tf.sqrt(tf.math.exp(sigma)))
    bce = 0.5*tf.keras.losses.binary_crossentropy(y_upper, c_pre) + 0.5*tf.keras.losses.binary_crossentropy(y_tr, c_pre) 
    l2 = (1/(tf.math.exp(temperature)))*(bce) + tf.math.log(tf.sqrt(tf.math.exp(temperature)))
    return tf.reduce_mean(l2)

def loss_MT_TPD_A(y_true, y_pred):
    #pri = y_true[:, :, :, 0]
    y_tr = tf.reshape(tf.reduce_max(y_true[:, :, :, 1], axis=(1, 2)), [-1, 1])
    y_upper = tf.reshape(tf.reduce_max(y_true[:, :, :, 2], axis=(1, 2)), [-1, 1])
    #d = tf.reshape(tf.reduce_max(y_true[:, :, :, 3], axis=(1, 2)), [-1, 1])
    d = tf.reduce_max(y_true[:, :, :, 3], axis=(1, 2))

    #pi_pre = y_pred[:, :, :, 0]
    c_pre = tf.reshape(tf.reduce_max(y_pred[:, :, :, 1], axis=(1, 2)), [-1, 1])
    #sigma = tf.reduce_max(y_pred[:, :, :, 2], axis=(1, 2))
    temperature = tf.reduce_max(y_pred[:, :, :, 3], axis=(1, 2))

    #l1 = (1/(2*tf.math.exp(sigma)))*tf.reduce_mean(tf.square(pi_pre - pri), axis=[1, 2]) + tf.math.log(tf.sqrt(tf.math.exp(sigma)))
    bce_r = tf.keras.losses.binary_crossentropy(y_tr, c_pre)
    bce_inst = tf.keras.losses.binary_crossentropy(c_pre, y_upper) 
    l2 = (1/(tf.math.exp(temperature)))*(0.5*(bce_r) + 0.5*(tf.math.multiply(d,bce_inst) - tf.math.multiply(1-d, bce_inst))) + tf.math.log(tf.sqrt(tf.math.exp(temperature)))
    return tf.reduce_mean(l2)




#(B) Epsilon = 0.  No regularization.

def loss_MT_B(y_true, y_pred):
    #pri = y_true[:, :, :, 0]
    y_tr = tf.reshape(tf.reduce_max(y_true[:, :, :, 1], axis=(1, 2)), [-1, 1])

    #pi_pre = y_pred[:, :, :, 0]
    c_pre = tf.reshape(tf.reduce_max(y_pred[:, :, :, 1], axis=(1, 2)), [-1, 1])
    #sigma = tf.reduce_max(y_pred[:, :, :, 2], axis=(1, 2))
    #temperature = tf.reduce_max(y_pred[:, :, :, 3], axis=(1, 2))

    #l1 = (1/(2*tf.math.exp(sigma)))*tf.reduce_mean(tf.square(pi_pre - pri), axis=[1, 2]) + tf.math.log(tf.sqrt(tf.math.exp(sigma)))
    l2 = tf.keras.losses.binary_crossentropy(y_tr, c_pre) 

    return tf.reduce_mean(l2)

def loss_MT_PFD_B(y_true, y_pred):
    #pri = y_true[:, :, :, 0]
    y_tr = tf.reshape(tf.reduce_max(y_true[:, :, :, 1], axis=(1, 2)), [-1, 1])
    y_upper = tf.reshape(tf.reduce_max(y_true[:, :, :, 2], axis=(1, 2)), [-1, 1])

    #pi_pre = y_pred[:, :, :, 0]
    c_pre = tf.reshape(tf.reduce_max(y_pred[:, :, :, 1], axis=(1, 2)), [-1, 1])
    #sigma = tf.reduce_max(y_pred[:, :, :, 2], axis=(1, 2))
    #temperature = tf.reduce_max(y_pred[:, :, :, 3], axis=(1, 2))

    #l1 = (1/(2*tf.math.exp(sigma)))*tf.reduce_mean(tf.square(pi_pre - pri), axis=[1, 2]) + tf.math.log(tf.sqrt(tf.math.exp(sigma)))
    bce = 0.5*tf.keras.losses.binary_crossentropy(y_upper, c_pre) + 0.5*tf.keras.losses.binary_crossentropy(y_tr, c_pre) 
    l2 = bce
    return tf.reduce_mean(l2)

def loss_MT_TPD_B(y_true, y_pred):
    #pri = y_true[:, :, :, 0]
    y_tr = tf.reshape(tf.reduce_max(y_true[:, :, :, 1], axis=(1, 2)), [-1, 1])
    y_upper = tf.reshape(tf.reduce_max(y_true[:, :, :, 2], axis=(1, 2)), [-1, 1])
    #d = tf.reshape(tf.reduce_max(y_true[:, :, :, 3], axis=(1, 2)), [-1, 1])
    d = tf.reduce_max(y_true[:, :, :, 3], axis=(1, 2))

    #pi_pre = y_pred[:, :, :, 0]
    c_pre = tf.reshape(tf.reduce_max(y_pred[:, :, :, 1], axis=(1, 2)), [-1, 1])
    #sigma = tf.reduce_max(y_pred[:, :, :, 2], axis=(1, 2))
    #temperature = tf.reduce_max(y_pred[:, :, :, 3], axis=(1, 2))

    #l1 = (1/(2*tf.math.exp(sigma)))*tf.reduce_mean(tf.square(pi_pre - pri), axis=[1, 2]) + tf.math.log(tf.sqrt(tf.math.exp(sigma)))
    bce_r = tf.keras.losses.binary_crossentropy(y_tr, c_pre)
    bce_inst = tf.keras.losses.binary_crossentropy(c_pre, y_upper) 
    l2 = (0.5*(bce_r) + 0.5*(tf.math.multiply(d,bce_inst) - tf.math.multiply(1-d, bce_inst)))
    return tf.reduce_mean(l2)