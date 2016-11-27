from __future__ import division
from baseagent import Agent
from collections import deque
from copy import deepcopy
import time
import numpy as np
import keras.backend as K
from keras.layers import Lambda, Input, merge, Layer
from keras.models import Model

from rl.core import Agent
from rl.policy import EpsGreedyQPolicy
from rl.util import *

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam


from pdb import set_trace as bp
def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))

"""
    Agent to look at only its own state
"""
class DQNAgent(Agent):

    def __init__(self,  id,teammates,opponents,actions,actionsEnum,inputV, nb_actions, memory, policy=EpsGreedyQPolicy(),
                 gamma=.99, batch_size=32, nb_steps_warmup=1000, train_interval=1, memory_interval=1,
                 target_model_update=10000, delta_range=(-np.inf, np.inf), enable_double_dqn=True,
                 custom_model_objects={}, processor=None):
        # Validate (important) input.

        self.model = self.createNetwork(inputV,nb_actions)
        if hasattr(self.model.output, '__len__') and len(self.model.output) > 1:
            raise ValueError('Model "{}" has more than one output. DQN expects a model that has a single output.'.format(model))
        if self.model.output._keras_shape != (None, nb_actions):
            raise ValueError('Model output "{}" has invalid shape. DQN expects a model that has one dimension for each action, in this case {}.'.format(model.output, nb_actions))

        super(DQNAgent, self).__init__()

        # Soft vs hard target model updates.
        if target_model_update < 0:
            raise ValueError('`target_model_update` must be >= 0.')
        elif target_model_update >= 1:
            # Hard update every `target_model_update` steps.
            target_model_update = int(target_model_update)
        else:
            # Soft update with `(1 - target_model_update) * old + target_model_update * new`.
            target_model_update = float(target_model_update)

        # Parameters.
        self.nb_actions = nb_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.nb_steps_warmup = nb_steps_warmup
        self.train_interval = train_interval
        self.memory_interval = memory_interval
        self.target_model_update = target_model_update
        self.delta_range = delta_range
        self.enable_double_dqn = enable_double_dqn
        self.custom_model_objects = custom_model_objects

        # Related objects.
        self.memory = memory
        self.policy = policy
        self.policy._set_agent(self)
        self.processor = processor

        # State.
        self.compiled = False
        self.reset_states()

        self.id=id
        teammates.remove(self.id)
        self.teammates= teammates
        self.opponents= opponents
        self.actions ,self.actionsEnum =actions,actionsEnum
        self.minAct,self.maxAct=min(self.actions),max(self.actions)


    def createNetwork(self,inputV,outputV):

        # this returns a tensor
        linputs = Input(shape=(11,))

        # a layer instance is callable on a tensor, and returns a tensor
        x = Dense(64, activation='relu')(linputs)
        x = Dense(64, activation='relu')(x)
        predictions = Dense(outputV, activation='softmax')(x)
        rinputs= Input(shape=(inputV-11,))
        y = Dense(64, activation='relu')(rinputs)
        xyu = merge([predictions, y], mode='concat')
        z=Dense(outputV, activation='softmax')(xyu)
        # this creates a model that includes
        # the Input layer and three Dense layers
        model = Model(input=[linputs,rinputs], output=z)
        # model = Sequential()
        # model.add(Flatten(input_shape=(1,) + (inputV,)))
        # model.add(Dense(16))
        # model.add(Activation('relu'))
        # model.add(Dense(16))
        # model.add(Activation('relu'))
        # model.add(Dense(16))
        # model.add(Activation('relu'))
        # model.add(Dense(outputV))
        # model.add(Activation('linear'))
        print model.summary()
        return model


    def get_config(self):
        config = {
            'nb_actions': self.nb_actions,
            'gamma': self.gamma,
            'batch_size': self.batch_size,
            'nb_steps_warmup': self.nb_steps_warmup,
            'train_interval': self.train_interval,
            'memory_interval': self.memory_interval,
            'target_model_update': self.target_model_update,
            'delta_range': self.delta_range,
            'enable_double_dqn': self.enable_double_dqn,
            'model': get_object_config(self.model),
            'memory': get_object_config(self.memory),
            'policy': get_object_config(self.policy),
        }
        if self.compiled:
            config['target_model'] = get_object_config(self.target_model)
        return config

    def compile(self, optimizer, metrics=[]):
        metrics += [mean_q]  # register default metrics

        # We never train the target model, hence we can set the optimizer and loss arbitrarily.
        self.target_model = clone_model(self.model, self.custom_model_objects)
        self.target_model.compile(optimizer='sgd', loss='mse')
        self.model.compile(optimizer='sgd', loss='mse')

        # Compile model.
        if self.target_model_update < 1.:
            # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
            updates = get_soft_target_model_updates(self.target_model, self.model, self.target_model_update)
            optimizer = AdditionalUpdatesOptimizer(optimizer, updates)

        def clipped_masked_mse(args):
            y_true, y_pred, mask = args
            delta = K.clip(y_true - y_pred, self.delta_range[0], self.delta_range[1])
            delta *= mask  # apply element-wise mask
            loss = K.mean(K.square(delta), axis=-1)
            # Multiply by the number of actions to reverse the effect of the mean.
            loss *= float(self.nb_actions)
            return loss

        # Create trainable model. The problem is that we need to mask the output since we only
        # ever want to update the Q values for a certain action. The way we achieve this is by
        # using a custom Lambda layer that computes the loss. This gives us the necessary flexibility
        # to mask out certain parameters by passing in multiple inputs to the Lambda layer.
        y_pred = self.model.output
        y_true = Input(name='y_true', shape=(self.nb_actions,))
        mask = Input(name='mask', shape=(self.nb_actions,))
        loss_out = Lambda(clipped_masked_mse, output_shape=(1,), name='loss')([y_pred, y_true, mask])

        trainable_model = Model(input=[self.model.input[0],self.model.input[1], y_true, mask], output=[loss_out, y_pred])
        assert len(trainable_model.output_names) == 2
        combined_metrics = {trainable_model.output_names[1]: metrics}
        losses = [
            lambda y_true, y_pred: y_pred,  # loss is computed in Lambda layer
            lambda y_true, y_pred: K.zeros_like(y_pred),  # we only include this for the metrics
        ]
        trainable_model.compile(optimizer=optimizer, loss=losses, metrics=combined_metrics)
        self.trainable_model = trainable_model

        self.compiled = True

    def load_weights(self, filepath):
        self.model.load_weights(filepath)
        self.update_target_model_hard()

    def save_weights(self, filepath, overwrite=False):
        self.model.save_weights(filepath, overwrite=overwrite)

    def reset_states(self):
        self.recent_action = None
        self.recent_observation = None
        if self.compiled:
            self.model.reset_states()
            self.target_model.reset_states()

    def update_target_model_hard(self):
        self.target_model.set_weights(self.model.get_weights())

    def process_state_batch(self, batch):
        batch = np.array(batch)
        if self.processor is None:
            return batch
        return self.processor.process_state_batch(batch)

    def compute_batch_q_values(self, state_batch):
        batch = self.process_state_batch(state_batch)
        z=[]
        z.append([0])
        b1=batch[0][0][:11].reshape((1,-1))
        #print "Shape of B1:"+str(b1.shape)
        b2=batch[0][0][11:].reshape((1,-1))
        #print "Shape of B2:"+str(b2.shape)
        z=[b1,b2]
        q_values = self.model.predict_on_batch(z)
        assert q_values.shape == (len(state_batch), self.nb_actions)
        return q_values

    def compute_q_values(self, state):
        q_values = self.compute_batch_q_values([state]).flatten()
        assert q_values.shape == (self.nb_actions,)
        return q_values

    def forward(self, observation):
        if self.processor is not None:
            observation = self.processor.process_observation(observation)

        # Select an action.
        state = self.memory.get_recent_state(observation)
        q_values = self.compute_q_values(state)
        action = self.policy.select_action(q_values=q_values)
        if self.processor is not None:
            action = self.processor.process_action(action)

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action

        return (action,q_values)

    def backward(self, reward, terminal):
        # Store most recent experience in memory.
        if self.processor is not None:
            reward = self.processor.process_reward(reward)
        if self.step % self.memory_interval == 0:
            self.memory.append(self.recent_observation, self.recent_action, reward, terminal,
                               training=self.training)

        metrics = [np.nan for _ in self.metrics_names]
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics

        # Train the network on a single stochastic batch.
        if self.step > self.nb_steps_warmup and self.step % self.train_interval == 0:
            experiences = self.memory.sample(self.batch_size)
            assert len(experiences) == self.batch_size

            # Start by extracting the necessary parameters (we use a vectorized implementation).
            state0_batch = []
            reward_batch = []
            action_batch = []
            terminal1_batch = []
            state1_batch = []
            for e in experiences:
                state0_batch.append(e.state0)
                state1_batch.append(e.state1)
                reward_batch.append(e.reward)
                action_batch.append(e.action)
                terminal1_batch.append(0. if e.terminal1 else 1.)

            # Prepare and validate parameters.
            state0_batch = self.process_state_batch(state0_batch)
            state1_batch = self.process_state_batch(state1_batch)
            terminal1_batch = np.array(terminal1_batch)
            reward_batch = np.array(reward_batch)
            assert reward_batch.shape == (self.batch_size,)
            assert terminal1_batch.shape == reward_batch.shape
            assert len(action_batch) == len(reward_batch)

            # Compute Q values for mini-batch update.
            if self.enable_double_dqn:
                # According to the paper "Deep Reinforcement Learning with Double Q-learning"
                # (van Hasselt et al., 2015), in Double DQN, the online network predicts the actions
                # while the target network is used to estimate the Q value.
                q_values = self.model.predict_on_batch(state1_batch)
                assert q_values.shape == (self.batch_size, self.nb_actions)
                actions = np.argmax(q_values, axis=1)
                assert actions.shape == (self.batch_size,)

                # Now, estimate Q values using the target network but select the values with the
                # highest Q value wrt to the online model (as computed above).
                target_q_values = self.target_model.predict_on_batch(state1_batch)
                assert target_q_values.shape == (self.batch_size, self.nb_actions)
                q_batch = target_q_values[range(self.batch_size), actions]
            else:
                # Compute the q_values given state1, and extract the maximum for each sample in the batch.
                # We perform this prediction on the target_model instead of the model for reasons
                # outlined in Mnih (2015). In short: it makes the algorithm more stable.
                target_q_values = self.target_model.predict_on_batch(state1_batch)
                assert target_q_values.shape == (self.batch_size, self.nb_actions)
                q_batch = np.max(target_q_values, axis=1).flatten()
            assert q_batch.shape == (self.batch_size,)

            targets = np.zeros((self.batch_size, self.nb_actions))
            dummy_targets = np.zeros((self.batch_size,))
            masks = np.zeros((self.batch_size, self.nb_actions))

            # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target targets accordingly,
            # but only for the affected output units (as given by action_batch).
            discounted_reward_batch = self.gamma * q_batch
            # Set discounted reward to zero for all states that were terminal.
            discounted_reward_batch *= terminal1_batch
            assert discounted_reward_batch.shape == reward_batch.shape
            Rs = reward_batch + discounted_reward_batch
            for idx, (target, mask, R, action) in enumerate(zip(targets, masks, Rs, action_batch)):
                target[action] = R  # update action with estimated accumulated reward
                dummy_targets[idx] = R
                mask[action] = 1.  # enable loss for this specific action
            targets = np.array(targets).astype('float32')
            masks = np.array(masks).astype('float32')

            # Finally, perform a single update on the entire batch. We use a dummy target since
            # the actual loss is computed in a Lambda layer that needs more complex input. However,
            # it is still useful to know the actual target to compute metrics properly.
            metrics = self.trainable_model.train_on_batch([state0_batch, targets, masks], [dummy_targets, targets])
            metrics = [metric for idx, metric in enumerate(metrics) if idx not in (1, 2)]  # throw away individual losses
            metrics += self.policy.metrics
            if self.processor is not None:
                metrics += self.processor.metrics

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_model_hard()

        return metrics

    @property
    def metrics_names(self):
        # Throw away individual losses and replace output name since this is hidden from the user.
        assert len(self.trainable_model.output_names) == 2
        dummy_output_name = self.trainable_model.output_names[1]
        model_metrics = [name for idx, name in enumerate(self.trainable_model.metrics_names) if idx not in (1, 2)]
        model_metrics = [name.replace(dummy_output_name + '_', '') for name in model_metrics]

        names = model_metrics + self.policy.metrics_names[:]
        if self.processor is not None:
            names += self.processor.metrics_names[:]
        return names

    def getAction(self,state):
        #print "Going In:"+str(state)
        try:
            (action,q_values) = self.forward(state)
        except:
            #print "Error occured"
            time.sleep(0.2)
            (action,q_values) = self.forward(state)
            #bp()

        print "Qvalue computed:-"+str(q_values)
        return action



    def perceive(self,agentState,teamState,opponentState,reward,terminal):
        self.backward(reward,terminal)

        #print "Perceiving"



class DQNTeamAgent(Agent):

    def __init__(self,  id,teammates,opponents,actions,actionsEnum,model, nb_actions, memory, policy=EpsGreedyQPolicy(),
                 gamma=.99, batch_size=32, nb_steps_warmup=1000, train_interval=1, memory_interval=1,
                 target_model_update=10000, delta_range=(-np.inf, np.inf), enable_double_dqn=True,
                 custom_model_objects={}, processor=None):
        # Validate (important) input.
        if hasattr(model.output, '__len__') and len(model.output) > 1:
            raise ValueError('Model "{}" has more than one output. DQN expects a model that has a single output.'.format(model))
        if model.output._keras_shape != (None, nb_actions):
            raise ValueError('Model output "{}" has invalid shape. DQN expects a model that has one dimension for each action, in this case {}.'.format(model.output, nb_actions))

        super(DQNAgent, self).__init__()

        # Soft vs hard target model updates.
        if target_model_update < 0:
            raise ValueError('`target_model_update` must be >= 0.')
        elif target_model_update >= 1:
            # Hard update every `target_model_update` steps.
            target_model_update = int(target_model_update)
        else:
            # Soft update with `(1 - target_model_update) * old + target_model_update * new`.
            target_model_update = float(target_model_update)

        # Parameters.
        self.nb_actions = nb_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.nb_steps_warmup = nb_steps_warmup
        self.train_interval = train_interval
        self.memory_interval = memory_interval
        self.target_model_update = target_model_update
        self.delta_range = delta_range
        self.enable_double_dqn = enable_double_dqn
        self.custom_model_objects = custom_model_objects

        # Related objects.
        self.model = model
        self.memory = memory
        self.policy = policy
        self.policy._set_agent(self)
        self.processor = processor

        # State.
        self.compiled = False
        self.reset_states()

        self.id=id
        teammates.remove(self.id)
        self.teammates= teammates
        self.opponents= opponents
        self.actions ,self.actionsEnum =actions,actionsEnum
        self.minAct,self.maxAct=min(self.actions),max(self.actions)

    def get_config(self):
        config = {
            'nb_actions': self.nb_actions,
            'gamma': self.gamma,
            'batch_size': self.batch_size,
            'nb_steps_warmup': self.nb_steps_warmup,
            'train_interval': self.train_interval,
            'memory_interval': self.memory_interval,
            'target_model_update': self.target_model_update,
            'delta_range': self.delta_range,
            'enable_double_dqn': self.enable_double_dqn,
            'model': get_object_config(self.model),
            'memory': get_object_config(self.memory),
            'policy': get_object_config(self.policy),
        }
        if self.compiled:
            config['target_model'] = get_object_config(self.target_model)
        return config

    def compile(self, optimizer, metrics=[]):
        metrics += [mean_q]  # register default metrics

        # We never train the target model, hence we can set the optimizer and loss arbitrarily.
        self.target_model = clone_model(self.model, self.custom_model_objects)
        self.target_model.compile(optimizer='sgd', loss='mse')
        self.model.compile(optimizer='sgd', loss='mse')

        # Compile model.
        if self.target_model_update < 1.:
            # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
            updates = get_soft_target_model_updates(self.target_model, self.model, self.target_model_update)
            optimizer = AdditionalUpdatesOptimizer(optimizer, updates)

        def clipped_masked_mse(args):
            y_true, y_pred, mask = args
            delta = K.clip(y_true - y_pred, self.delta_range[0], self.delta_range[1])
            delta *= mask  # apply element-wise mask
            loss = K.mean(K.square(delta), axis=-1)
            # Multiply by the number of actions to reverse the effect of the mean.
            loss *= float(self.nb_actions)
            return loss

        # Create trainable model. The problem is that we need to mask the output since we only
        # ever want to update the Q values for a certain action. The way we achieve this is by
        # using a custom Lambda layer that computes the loss. This gives us the necessary flexibility
        # to mask out certain parameters by passing in multiple inputs to the Lambda layer.
        y_pred = self.model.output
        y_true = Input(name='y_true', shape=(self.nb_actions,))
        mask = Input(name='mask', shape=(self.nb_actions,))
        loss_out = Lambda(clipped_masked_mse, output_shape=(1,), name='loss')([y_pred, y_true, mask])
        trainable_model = Model(input=[self.model.input, y_true, mask], output=[loss_out, y_pred])
        assert len(trainable_model.output_names) == 2
        combined_metrics = {trainable_model.output_names[1]: metrics}
        losses = [
            lambda y_true, y_pred: y_pred,  # loss is computed in Lambda layer
            lambda y_true, y_pred: K.zeros_like(y_pred),  # we only include this for the metrics
        ]
        trainable_model.compile(optimizer=optimizer, loss=losses, metrics=combined_metrics)
        self.trainable_model = trainable_model

        self.compiled = True

    def load_weights(self, filepath):
        self.model.load_weights(filepath)
        self.update_target_model_hard()

    def save_weights(self, filepath, overwrite=False):
        self.model.save_weights(filepath, overwrite=overwrite)

    def reset_states(self):
        self.recent_action = None
        self.recent_observation = None
        if self.compiled:
            self.model.reset_states()
            self.target_model.reset_states()

    def update_target_model_hard(self):
        self.target_model.set_weights(self.model.get_weights())

    def process_state_batch(self, batch):
        batch = np.array(batch)
        if self.processor is None:
            return batch
        return self.processor.process_state_batch(batch)

    def compute_batch_q_values(self, state_batch):
        batch = self.process_state_batch(state_batch)
        q_values = self.model.predict_on_batch(batch)
        assert q_values.shape == (len(state_batch), self.nb_actions)
        return q_values

    def compute_q_values(self, state):
        q_values = self.compute_batch_q_values([state]).flatten()
        assert q_values.shape == (self.nb_actions,)
        return q_values

    def forward(self, observation):
        if self.processor is not None:
            observation = self.processor.process_observation(observation)

        # Select an action.
        state = self.memory.get_recent_state(observation)
        q_values = self.compute_q_values(state)
        action = self.policy.select_action(q_values=q_values)
        if self.processor is not None:
            action = self.processor.process_action(action)

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action

        return (action,q_values)

    def backward(self, reward, terminal):
        # Store most recent experience in memory.
        if self.processor is not None:
            reward = self.processor.process_reward(reward)
        if self.step % self.memory_interval == 0:
            self.memory.append(self.recent_observation, self.recent_action, reward, terminal,
                               training=self.training)

        metrics = [np.nan for _ in self.metrics_names]
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics

        # Train the network on a single stochastic batch.
        if self.step > self.nb_steps_warmup and self.step % self.train_interval == 0:
            experiences = self.memory.sample(self.batch_size)
            assert len(experiences) == self.batch_size

            # Start by extracting the necessary parameters (we use a vectorized implementation).
            state0_batch = []
            reward_batch = []
            action_batch = []
            terminal1_batch = []
            state1_batch = []
            for e in experiences:
                state0_batch.append(e.state0)
                state1_batch.append(e.state1)
                reward_batch.append(e.reward)
                action_batch.append(e.action)
                terminal1_batch.append(0. if e.terminal1 else 1.)

            # Prepare and validate parameters.
            state0_batch = self.process_state_batch(state0_batch)
            state1_batch = self.process_state_batch(state1_batch)
            terminal1_batch = np.array(terminal1_batch)
            reward_batch = np.array(reward_batch)
            assert reward_batch.shape == (self.batch_size,)
            assert terminal1_batch.shape == reward_batch.shape
            assert len(action_batch) == len(reward_batch)

            # Compute Q values for mini-batch update.
            if self.enable_double_dqn:
                # According to the paper "Deep Reinforcement Learning with Double Q-learning"
                # (van Hasselt et al., 2015), in Double DQN, the online network predicts the actions
                # while the target network is used to estimate the Q value.
                q_values = self.model.predict_on_batch(state1_batch)
                assert q_values.shape == (self.batch_size, self.nb_actions)
                actions = np.argmax(q_values, axis=1)
                assert actions.shape == (self.batch_size,)

                # Now, estimate Q values using the target network but select the values with the
                # highest Q value wrt to the online model (as computed above).
                target_q_values = self.target_model.predict_on_batch(state1_batch)
                assert target_q_values.shape == (self.batch_size, self.nb_actions)
                q_batch = target_q_values[range(self.batch_size), actions]
            else:
                # Compute the q_values given state1, and extract the maximum for each sample in the batch.
                # We perform this prediction on the target_model instead of the model for reasons
                # outlined in Mnih (2015). In short: it makes the algorithm more stable.
                target_q_values = self.target_model.predict_on_batch(state1_batch)
                assert target_q_values.shape == (self.batch_size, self.nb_actions)
                q_batch = np.max(target_q_values, axis=1).flatten()
            assert q_batch.shape == (self.batch_size,)

            targets = np.zeros((self.batch_size, self.nb_actions))
            dummy_targets = np.zeros((self.batch_size,))
            masks = np.zeros((self.batch_size, self.nb_actions))

            # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target targets accordingly,
            # but only for the affected output units (as given by action_batch).
            discounted_reward_batch = self.gamma * q_batch
            # Set discounted reward to zero for all states that were terminal.
            discounted_reward_batch *= terminal1_batch
            assert discounted_reward_batch.shape == reward_batch.shape
            Rs = reward_batch + discounted_reward_batch
            for idx, (target, mask, R, action) in enumerate(zip(targets, masks, Rs, action_batch)):
                target[action] = R  # update action with estimated accumulated reward
                dummy_targets[idx] = R
                mask[action] = 1.  # enable loss for this specific action
            targets = np.array(targets).astype('float32')
            masks = np.array(masks).astype('float32')

            # Finally, perform a single update on the entire batch. We use a dummy target since
            # the actual loss is computed in a Lambda layer that needs more complex input. However,
            # it is still useful to know the actual target to compute metrics properly.
            metrics = self.trainable_model.train_on_batch([state0_batch, targets, masks], [dummy_targets, targets])
            metrics = [metric for idx, metric in enumerate(metrics) if idx not in (1, 2)]  # throw away individual losses
            metrics += self.policy.metrics
            if self.processor is not None:
                metrics += self.processor.metrics

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_model_hard()

        return metrics

    @property
    def metrics_names(self):
        # Throw away individual losses and replace output name since this is hidden from the user.
        assert len(self.trainable_model.output_names) == 2
        dummy_output_name = self.trainable_model.output_names[1]
        model_metrics = [name for idx, name in enumerate(self.trainable_model.metrics_names) if idx not in (1, 2)]
        model_metrics = [name.replace(dummy_output_name + '_', '') for name in model_metrics]

        names = model_metrics + self.policy.metrics_names[:]
        if self.processor is not None:
            names += self.processor.metrics_names[:]
        return names

    def getAction(self,state):
        #print "Going In:"+str(state)
        try:
            action = self.forward(state)
        except:
            #print "Error occured"
            time.sleep(0.2)
            action = self.forward(state)
            #bp()
        return action



    def perceive(self,agentState,teamState,opponentState,reward,terminal):
        self.backward(reward,terminal)

        #print "Perceiving"









# class DQNAgent(Agent):
#
#     def __init__(self,  id,teammates,opponents,actions,actionsEnum,model, nb_actions, memory, policy=EpsGreedyQPolicy(),
#                  gamma=.99, batch_size=32, nb_steps_warmup=1000, train_interval=1, memory_interval=1,
#                  target_model_update=10000, delta_range=(-np.inf, np.inf), enable_double_dqn=True,
#                  custom_model_objects={}, processor=None):
#         # Validate (important) input.
#         if hasattr(model.output, '__len__') and len(model.output) > 1:
#             raise ValueError('Model "{}" has more than one output. DQN expects a model that has a single output.'.format(model))
#         if model.output._keras_shape != (None, nb_actions):
#             raise ValueError('Model output "{}" has invalid shape. DQN expects a model that has one dimension for each action, in this case {}.'.format(model.output, nb_actions))
#
#         super(DQNAgent, self).__init__()
#
#         # Soft vs hard target model updates.
#         if target_model_update < 0:
#             raise ValueError('`target_model_update` must be >= 0.')
#         elif target_model_update >= 1:
#             # Hard update every `target_model_update` steps.
#             target_model_update = int(target_model_update)
#         else:
#             # Soft update with `(1 - target_model_update) * old + target_model_update * new`.
#             target_model_update = float(target_model_update)
#
#         # Parameters.
#         self.nb_actions = nb_actions
#         self.gamma = gamma
#         self.batch_size = batch_size
#         self.nb_steps_warmup = nb_steps_warmup
#         self.train_interval = train_interval
#         self.memory_interval = memory_interval
#         self.target_model_update = target_model_update
#         self.delta_range = delta_range
#         self.enable_double_dqn = enable_double_dqn
#         self.custom_model_objects = custom_model_objects
#
#         # Related objects.
#         self.model = model
#         self.memory = memory
#         self.policy = policy
#         self.policy._set_agent(self)
#         self.processor = processor
#
#         # State.
#         self.compiled = False
#         self.reset_states()
#
#         self.id=id
#         teammates.remove(self.id)
#         self.teammates= teammates
#         self.opponents= opponents
#         self.actions ,self.actionsEnum =actions,actionsEnum
#         self.minAct,self.maxAct=min(self.actions),max(self.actions)
#
#     def get_config(self):
#         config = {
#             'nb_actions': self.nb_actions,
#             'gamma': self.gamma,
#             'batch_size': self.batch_size,
#             'nb_steps_warmup': self.nb_steps_warmup,
#             'train_interval': self.train_interval,
#             'memory_interval': self.memory_interval,
#             'target_model_update': self.target_model_update,
#             'delta_range': self.delta_range,
#             'enable_double_dqn': self.enable_double_dqn,
#             'model': get_object_config(self.model),
#             'memory': get_object_config(self.memory),
#             'policy': get_object_config(self.policy),
#         }
#         if self.compiled:
#             config['target_model'] = get_object_config(self.target_model)
#         return config
#
#     def compile(self, optimizer, metrics=[]):
#         metrics += [mean_q]  # register default metrics
#
#         # We never train the target model, hence we can set the optimizer and loss arbitrarily.
#         self.target_model = clone_model(self.model, self.custom_model_objects)
#         self.target_model.compile(optimizer='sgd', loss='mse')
#         self.model.compile(optimizer='sgd', loss='mse')
#
#         # Compile model.
#         if self.target_model_update < 1.:
#             # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
#             updates = get_soft_target_model_updates(self.target_model, self.model, self.target_model_update)
#             optimizer = AdditionalUpdatesOptimizer(optimizer, updates)
#
#         def clipped_masked_mse(args):
#             y_true, y_pred, mask = args
#             delta = K.clip(y_true - y_pred, self.delta_range[0], self.delta_range[1])
#             delta *= mask  # apply element-wise mask
#             loss = K.mean(K.square(delta), axis=-1)
#             # Multiply by the number of actions to reverse the effect of the mean.
#             loss *= float(self.nb_actions)
#             return loss
#
#         # Create trainable model. The problem is that we need to mask the output since we only
#         # ever want to update the Q values for a certain action. The way we achieve this is by
#         # using a custom Lambda layer that computes the loss. This gives us the necessary flexibility
#         # to mask out certain parameters by passing in multiple inputs to the Lambda layer.
#         y_pred = self.model.output
#         y_true = Input(name='y_true', shape=(self.nb_actions,))
#         mask = Input(name='mask', shape=(self.nb_actions,))
#         loss_out = Lambda(clipped_masked_mse, output_shape=(1,), name='loss')([y_pred, y_true, mask])
#         trainable_model = Model(input=[self.model.input, y_true, mask], output=[loss_out, y_pred])
#         assert len(trainable_model.output_names) == 2
#         combined_metrics = {trainable_model.output_names[1]: metrics}
#         losses = [
#             lambda y_true, y_pred: y_pred,  # loss is computed in Lambda layer
#             lambda y_true, y_pred: K.zeros_like(y_pred),  # we only include this for the metrics
#         ]
#         trainable_model.compile(optimizer=optimizer, loss=losses, metrics=combined_metrics)
#         self.trainable_model = trainable_model
#
#         self.compiled = True
#
#     def load_weights(self, filepath):
#         self.model.load_weights(filepath)
#         self.update_target_model_hard()
#
#     def save_weights(self, filepath, overwrite=False):
#         self.model.save_weights(filepath, overwrite=overwrite)
#
#     def reset_states(self):
#         self.recent_action = None
#         self.recent_observation = None
#         if self.compiled:
#             self.model.reset_states()
#             self.target_model.reset_states()
#
#     def update_target_model_hard(self):
#         self.target_model.set_weights(self.model.get_weights())
#
#     def process_state_batch(self, batch):
#         batch = np.array(batch)
#         if self.processor is None:
#             return batch
#         return self.processor.process_state_batch(batch)
#
#     def compute_batch_q_values(self, state_batch):
#         batch = self.process_state_batch(state_batch)
#         q_values = self.model.predict_on_batch(batch)
#         assert q_values.shape == (len(state_batch), self.nb_actions)
#         return q_values
#
#     def compute_q_values(self, state):
#         q_values = self.compute_batch_q_values([state]).flatten()
#         assert q_values.shape == (self.nb_actions,)
#         return q_values
#
#     def forward(self, observation):
#         if self.processor is not None:
#             observation = self.processor.process_observation(observation)
#
#         # Select an action.
#         state = self.memory.get_recent_state(observation)
#         q_values = self.compute_q_values(state)
#         action = self.policy.select_action(q_values=q_values)
#         if self.processor is not None:
#             action = self.processor.process_action(action)
#
#         # Book-keeping.
#         self.recent_observation = observation
#         self.recent_action = action
#
#         return action
#
#     def backward(self, reward, terminal):
#         # Store most recent experience in memory.
#         if self.processor is not None:
#             reward = self.processor.process_reward(reward)
#         if self.step % self.memory_interval == 0:
#             self.memory.append(self.recent_observation, self.recent_action, reward, terminal,
#                                training=self.training)
#
#         metrics = [np.nan for _ in self.metrics_names]
#         if not self.training:
#             # We're done here. No need to update the experience memory since we only use the working
#             # memory to obtain the state over the most recent observations.
#             return metrics
#
#         # Train the network on a single stochastic batch.
#         if self.step > self.nb_steps_warmup and self.step % self.train_interval == 0:
#             experiences = self.memory.sample(self.batch_size)
#             assert len(experiences) == self.batch_size
#
#             # Start by extracting the necessary parameters (we use a vectorized implementation).
#             state0_batch = []
#             reward_batch = []
#             action_batch = []
#             terminal1_batch = []
#             state1_batch = []
#             for e in experiences:
#                 state0_batch.append(e.state0)
#                 state1_batch.append(e.state1)
#                 reward_batch.append(e.reward)
#                 action_batch.append(e.action)
#                 terminal1_batch.append(0. if e.terminal1 else 1.)
#
#             # Prepare and validate parameters.
#             state0_batch = self.process_state_batch(state0_batch)
#             state1_batch = self.process_state_batch(state1_batch)
#             terminal1_batch = np.array(terminal1_batch)
#             reward_batch = np.array(reward_batch)
#             assert reward_batch.shape == (self.batch_size,)
#             assert terminal1_batch.shape == reward_batch.shape
#             assert len(action_batch) == len(reward_batch)
#
#             # Compute Q values for mini-batch update.
#             if self.enable_double_dqn:
#                 # According to the paper "Deep Reinforcement Learning with Double Q-learning"
#                 # (van Hasselt et al., 2015), in Double DQN, the online network predicts the actions
#                 # while the target network is used to estimate the Q value.
#                 q_values = self.model.predict_on_batch(state1_batch)
#                 assert q_values.shape == (self.batch_size, self.nb_actions)
#                 actions = np.argmax(q_values, axis=1)
#                 assert actions.shape == (self.batch_size,)
#
#                 # Now, estimate Q values using the target network but select the values with the
#                 # highest Q value wrt to the online model (as computed above).
#                 target_q_values = self.target_model.predict_on_batch(state1_batch)
#                 assert target_q_values.shape == (self.batch_size, self.nb_actions)
#                 q_batch = target_q_values[range(self.batch_size), actions]
#             else:
#                 # Compute the q_values given state1, and extract the maximum for each sample in the batch.
#                 # We perform this prediction on the target_model instead of the model for reasons
#                 # outlined in Mnih (2015). In short: it makes the algorithm more stable.
#                 target_q_values = self.target_model.predict_on_batch(state1_batch)
#                 assert target_q_values.shape == (self.batch_size, self.nb_actions)
#                 q_batch = np.max(target_q_values, axis=1).flatten()
#             assert q_batch.shape == (self.batch_size,)
#
#             targets = np.zeros((self.batch_size, self.nb_actions))
#             dummy_targets = np.zeros((self.batch_size,))
#             masks = np.zeros((self.batch_size, self.nb_actions))
#
#             # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target targets accordingly,
#             # but only for the affected output units (as given by action_batch).
#             discounted_reward_batch = self.gamma * q_batch
#             # Set discounted reward to zero for all states that were terminal.
#             discounted_reward_batch *= terminal1_batch
#             assert discounted_reward_batch.shape == reward_batch.shape
#             Rs = reward_batch + discounted_reward_batch
#             for idx, (target, mask, R, action) in enumerate(zip(targets, masks, Rs, action_batch)):
#                 target[action] = R  # update action with estimated accumulated reward
#                 dummy_targets[idx] = R
#                 mask[action] = 1.  # enable loss for this specific action
#             targets = np.array(targets).astype('float32')
#             masks = np.array(masks).astype('float32')
#
#             # Finally, perform a single update on the entire batch. We use a dummy target since
#             # the actual loss is computed in a Lambda layer that needs more complex input. However,
#             # it is still useful to know the actual target to compute metrics properly.
#             metrics = self.trainable_model.train_on_batch([state0_batch, targets, masks], [dummy_targets, targets])
#             metrics = [metric for idx, metric in enumerate(metrics) if idx not in (1, 2)]  # throw away individual losses
#             metrics += self.policy.metrics
#             if self.processor is not None:
#                 metrics += self.processor.metrics
#
#         if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
#             self.update_target_model_hard()
#
#         return metrics
#
#     @property
#     def metrics_names(self):
#         # Throw away individual losses and replace output name since this is hidden from the user.
#         assert len(self.trainable_model.output_names) == 2
#         dummy_output_name = self.trainable_model.output_names[1]
#         model_metrics = [name for idx, name in enumerate(self.trainable_model.metrics_names) if idx not in (1, 2)]
#         model_metrics = [name.replace(dummy_output_name + '_', '') for name in model_metrics]
#
#         names = model_metrics + self.policy.metrics_names[:]
#         if self.processor is not None:
#             names += self.processor.metrics_names[:]
#         return names
#
#     def getAction(self,state):
#         #print "Going In:"+str(state)
#         try:
#             action = self.forward(state)
#         except:
#             #print "Error occured"
#             time.sleep(0.2)
#             action = self.forward(state)
#             #bp()
#         return action
#
#
#
#     def perceive(self,agentState,teamState,opponentState,reward,terminal):
#         self.backward(reward,terminal)
#
#         #print "Perceiving"
