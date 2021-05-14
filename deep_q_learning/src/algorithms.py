import tensorflow as tf
from tensorflow.keras.losses import MSE, Huber


huber_loss = Huber()


@tf.function
def train_dqn(
    model, model_target, optimizer, discount_factor, num_actions,
    state_sample, action_sample, rewards_sample, state_next_sample, done_sample):
    rewards_sample = tf.cast(rewards_sample, tf.float32)
    done_sample = tf.cast(done_sample, tf.float32)

    with tf.GradientTape() as tape:
        q_values = model(state_sample)
        q_action = tf.gather(q_values, action_sample, batch_dims=1)
        with tape.stop_recording():
            future_rewards = model_target(state_next_sample) * (1.0 - done_sample)
            q_values_target = rewards_sample + discount_factor * tf.reduce_max(future_rewards, axis=1, keepdims=True)
        loss = huber_loss(q_values_target, q_action)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, tf.reduce_max(q_values)


@tf.function
def train_ddqn(
    model, model_target, optimizer, discount_factor, num_actions,
    state_sample, action_sample, rewards_sample, state_next_sample, done_sample):
    rewards_sample = tf.cast(rewards_sample, tf.float32)
    done_sample = tf.cast(done_sample, tf.float32)

    with tf.GradientTape() as tape:
        action_next = tf.math.argmax(model(state_next_sample), axis=1)
        action_next = tf.expand_dims(action_next, axis=1)
        with tape.stop_recording():
            q_values_next = model_target(state_next_sample)
            future_rewards = tf.gather(q_values_next, action_next, batch_dims=1) * (1 - done_sample)
            q_values_target = rewards_sample + discount_factor * future_rewards
        q_values = model(state_sample)
        q_action = tf.gather(q_values, action_sample, batch_dims=1)
        loss = huber_loss(q_values_target, q_action)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, tf.reduce_max(q_values)
