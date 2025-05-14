import tensorflow as tf

def build_neural_network(X,Y1,Y2,Y3,Y4):
    #Convert from pandas to tensor
    X_tensor = tf.convert_to_tensor(X.values,dtype=tf.float32)
    Y1_tensor =tf.convert_to_tensor(Y1.values,dtype=tf.float32)
    Y2_tensor = tf.convert_to_tensor(Y2.values, dtype=tf.float32)
    Y3_tensor = tf.convert_to_tensor(Y3.values, dtype=tf.float32)
    Y4_tensor = tf.convert_to_tensor(Y4.values, dtype=tf.float32)

    #Input layer
    inp = tf.keras.Input(shape=(5,))

    #Hidden Layers
    hidden1 = tf.keras.layers.Dense(128, activation= 'relu')(inp)
    hidden2 = tf.keras.layers.Dense(64, activation='relu')(hidden1)
    hidden3 = tf.keras.layers.Dense(32, activation='relu')(hidden2)
    hidden4 = tf.keras.layers.Dense(16, activation='relu')(hidden3)

    #Output Layers
    out1 = tf.keras.layers.Dense(1, activation ='linear')(hidden4)
    out2 = tf.keras.layers.Dense(1, activation='linear')(hidden4)
    out3 = tf.keras.layers.Dense(1, activation='linear')(hidden4)
    out4 = tf.keras.layers.Dense(1, activation='linear')(hidden4)

    #Model
    model = tf.keras.Model(inp, [out1,out2,out3,out4])
    return model

