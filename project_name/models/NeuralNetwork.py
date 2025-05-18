import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def build_neural_network(X,Y1,Y2,Y3,Y4):
    #print statements to check inbalanced classes
    print(Y1.value_counts(normalize=True).round(3))
    print(Y2.value_counts(normalize=True).round(3))
    print(Y3.value_counts(normalize=True).round(3))
    print(Y4.value_counts(normalize=True).round(3))

    #splitting in test and train data
    X_train,X_test,Y1_train,Y1_test,Y2_train,Y2_test,Y3_train,Y3_test,Y4_train,Y4_test = train_test_split(X,Y1,Y2,Y3,Y4,test_size=0.2)

    #Normalisation of the x-features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #Convert the train data from pandas to tensor
    X_tensor = tf.convert_to_tensor(X_train,dtype=tf.float32)
    Y1_tensor =tf.convert_to_tensor(Y1_train.values,dtype=tf.int32)
    Y2_tensor = tf.convert_to_tensor(Y2_train.values-1, dtype=tf.int32)
    Y3_tensor = tf.convert_to_tensor(Y3_train.values-1, dtype=tf.int32)
    Y4_tensor = tf.convert_to_tensor(Y4_train.values-1, dtype=tf.int32)

    #Input Layer
    inp = tf.keras.Input(shape=(5,))

    #Hidden Layers
    hidden1 = tf.keras.layers.Dense(128, activation= 'relu')(inp)
    hidden2 = tf.keras.layers.Dense(64, activation='relu')(hidden1)
    hidden3 = tf.keras.layers.Dense(32, activation='relu')(hidden2)
    hidden4 = tf.keras.layers.Dense(16, activation='relu')(hidden3)

    #Output Layers
    out1 = tf.keras.layers.Dense(19, activation ='softmax', name="Social_media")(hidden4)
    out2 = tf.keras.layers.Dense(5, activation='softmax',name='think_body')(hidden4)
    out3 = tf.keras.layers.Dense(5, activation='softmax', name='feeling_low')(hidden4)
    out4 = tf.keras.layers.Dense(5, activation='softmax', name='sleep_difficulty')(hidden4)

    #Model
    model = tf.keras.Model(inp, [out1,out2,out3,out4])
    model.compile(
        optimizer='adam',
        loss={
            'Social_media': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            'think_body': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            'feeling_low': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            'sleep_difficulty': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        },
        metrics={
            'Social_media': tf.keras.metrics.SparseCategoricalAccuracy(),
            'think_body': tf.keras.metrics.SparseCategoricalAccuracy(),
            'feeling_low': tf.keras.metrics.SparseCategoricalAccuracy(),
            'sleep_difficulty': tf.keras.metrics.SparseCategoricalAccuracy()
        }
    )
    model.fit(X_tensor, {
        'Social_media': Y1_tensor,
        'think_body': Y2_tensor,
        'feeling_low': Y3_tensor,
        'sleep_difficulty': Y4_tensor
        }, epochs=10, batch_size=32, validation_split=0.2
    )
    return model
