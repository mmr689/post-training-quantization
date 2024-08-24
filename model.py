import tensorflow as tf

model = tf.keras.applications.MobileNet(weights='imagenet')

tf.saved_model.save(model, "PIUsatge")