import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import keras

def build_neural_network(X, Y1, Y2, Y3,input_size):
    #print statements to check inbalanced classes
    print(Y1.value_counts(normalize=True).round(3))
    print(Y2.value_counts(normalize=True).round(3))
    print(Y3.value_counts(normalize=True).round(3))

    #splitting in test and train data
    X_train,X_test,Y1_train,Y1_test,Y2_train,Y2_test,Y3_train,Y3_test = train_test_split(X, Y1, Y2, Y3, test_size=.2)

    #Print statement for checking with and without using stratify
    print('Class distribution for the train set of Y3')
    print(Y1_train.value_counts(normalize=True).round(3))
    print('Class distribution for the test set of Y3')
    print(Y1_test.value_counts(normalize=True).round(3))

    #Normalisation of the x-features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #Convert the train data from pandas to tensor
    X_tensor = tf.convert_to_tensor(X_train,dtype=tf.float32)
    Y1_tensor = tf.convert_to_tensor(Y1_train.values-1, dtype=tf.int32)
    Y2_tensor = tf.convert_to_tensor(Y2_train.values-1, dtype=tf.int32)
    Y3_tensor = tf.convert_to_tensor(Y3_train.values-1, dtype=tf.int32)

    #Input Layer
    inp = tf.keras.Input(shape=(input_size,))

    #Hidden Layers
    hidden1 = tf.keras.layers.Dense(128, activation= 'relu')(inp)
    hidden2 = tf.keras.layers.Dense(64, activation='relu')(hidden1)
    hidden3 = tf.keras.layers.Dense(32, activation='relu')(hidden2)
    hidden4 = tf.keras.layers.Dense(16, activation='relu')(hidden3)

    #Output Layers
    out1 = tf.keras.layers.Dense(5, activation='softmax',name='think_body')(hidden4)
    out2 = tf.keras.layers.Dense(5, activation='softmax', name='feeling_low')(hidden4)
    out3 = tf.keras.layers.Dense(5, activation='softmax', name='sleep_difficulty')(hidden4)

    #Model
    model = tf.keras.Model(inp, [out1,out2,out3])
    model.compile(
        optimizer='adam',
        loss={
            'think_body': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            'feeling_low': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            'sleep_difficulty': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        },
        metrics={
            'think_body': tf.keras.metrics.SparseCategoricalAccuracy(),
            'feeling_low': tf.keras.metrics.SparseCategoricalAccuracy(),
            'sleep_difficulty': tf.keras.metrics.SparseCategoricalAccuracy()
        }
    )
    model.fit(X_tensor, {
        'think_body': Y1_tensor,
        'feeling_low': Y2_tensor,
        'sleep_difficulty': Y3_tensor
        }, epochs=1, batch_size=32, validation_split=0.2
    )
    #Evaluating the model on the test data
    #results = model.evaluate(X_test,Y1_test,Y2_test,Y3_test)
    
    #convert to tensor
    #model = model(tf.convert_to_tensor(X_train[:1].astype(np.float32)))
    
    return model, X_train, X_test, scaler #X_train, X_test and scaler are remembered for SHAP
