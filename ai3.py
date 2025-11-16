import tensorflow as tf

snack_tensor = tf.constant([7, 5, 6, 4], dtype=tf.float32)
weights_tensor = tf.constant([0.4, 0.3, 0.2, 0.1], dtype=tf.float32)

weighted_features = tf.multiply(snack_tensor, weights_tensor)
sum_features = tf.reduce_sum(weighted_features)

activated_features = tf.nn.relu(sum_features)
snack_rating = tf.round(activated_features)

tf.print("Рейтинг привлекательности:", snack_rating)