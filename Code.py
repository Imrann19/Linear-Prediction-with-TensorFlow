import tensorflow as tf
import random


model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])#Construction du modèle / Building the model

model.compile(optimizer = 'sgd', loss = 'mean_squared_error')#Compilation du modèle / Compiling the model

#Donnés du modèle / Datasets of the model
x_train = [1, 2, 3, 4, 5]#L'entrée du modèle / model input
y_train = [2, 4, 6, 8, 10]#La sortie du modèle / model output

model.fit(x_train, y_train, epochs=10, verbose=0)

X = random.randint(1, 1000)#Nombre aléatoire(entrée du modèle) / random number(model input)

#Affichage des résultats / Displaying of results
print(model.predict([X])) 
print(f"\nLe nombre réel est {X} et le nombre à prédire était : {X*2}" )
