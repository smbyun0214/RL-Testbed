from tensorflow.keras.models import clone_model


class Agent(object):
    def __init__(self,
        model,
        num_actions,
        behavior_policy,
        train,
        optimizer,
        epsilon_init=1.0,
        epsilon_fin=0.1,
        epsilon_fin_frame=float(1e+6),
        ):
        self._model = model
        self._model_target = clone_model(model)
        self._model_target.set_weights(self._model.get_weights())
       
        self._num_actions = num_actions
       
        self._behavior = behavior_policy
        self._train = train

        self.epsilon = epsilon_init
        self._epsilon_fin = epsilon_fin
        self._epsilon_fin_frame = epsilon_fin_frame
        self._epsilon_interval = epsilon_init - epsilon_fin

        self._optimizer = optimizer

    def get_action(self, state, epsilon=None):
        if epsilon is None:
            act = self._behavior(self._model, state, self.epsilon)
            self.epsilon -= self._epsilon_interval / self._epsilon_fin_frame
            self.epsilon = max(self.epsilon, self._epsilon_fin)
        else:
            act = self._behavior(self._model, state, 1.0)
        return act

    def train(self, state_sample, action_sample, state_next_sample, rewards_sample, done_sample, discount_factor=0.99):
        loss, q_max = self._train(
            self._model, self._model_target, self._optimizer,
            discount_factor, self._num_actions,
            state_sample, action_sample, rewards_sample, state_next_sample, done_sample)
        return loss, q_max

    def update_model_target(self):
        self._model_target.set_weights(self._model.get_weights())

    def save_model(self, path):
        self._model.save_weights(path)
