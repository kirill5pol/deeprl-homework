from __future__ import absolute_import, print_function, division

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import random
import pickle
import math
import gym
import sys
import os

from collections import namedtuple
from itertools import count
from PIL import Image

import tensorflow as tf
import tf_util as tfu
import seaborn as sns
import numpy as np
import load_policy


class Policy(tf.keras.Model):
    def __init__(self, num_outputs=3, layers=[400, 200], dropout_rate=.2, activation=tf.nn.relu):
        super(Policy, self).__init__()
        kernel_initializer = tf.initializers.variance_scaling(scale=2.0)

        self._layers = []
        self._dropouts = []
        for size in layers:
            self._dropouts.append(tf.layers.Dropout(rate=dropout_rate))
            self._layers.append(
                tf.layers.Dense(
                    units=size, activation=activation,
                    kernel_initializer=kernel_initializer))
        self.logits = tf.layers.Dense(
            units=num_outputs, activation=None, 
            kernel_initializer=kernel_initializer)

    def call(self, inputs, training=False):
        hidden = inputs
        for dropout, layer in zip(self._dropouts, self._layers[0]):
            hidden = dropout(layer(hidden), training=training)
        logits = self.logits(hidden)
        return logits


def initial_dataset(envname):
    '''Return the intial expert data from an existing file.'''
    expert_data = None
    filename = 'expert_data/' + envname + '.pkl'
    with open(filename, 'rb') as f:
        expert_data = pickle.load(f)
    return expert_data['observations'], expert_data['actions']


def expert_labels(policy_fn, observations):
    observations = np.array(observations)
    observations = observations.reshape((observations.shape[0], -1))
    actions = policy_fn(observations)

    return actions


def imitator_rollouts(sess, env, input_ph, output_pred, max_timesteps=None, num_rollouts=20, render=False):
    max_steps = max_timesteps or env.spec.timestep_limit

    returns = []
    observations = []
    actions = []

    for i in range(num_rollouts):
        observation = env.reset()

        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            obs = obs.reshape((1, -1))
            action = sess.run([output_pred], feed_dict={input_ph: obs})
            action = action[0].tolist()[0]
            obs = obs.tolist()[0]
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if render:
                env.render()
            if steps % 100 == 0:
                print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)

    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

    return returns, observations, actions


def create_imitator_policy(input_shape=11, output_shape=3, layers=[400, 400, 400], dropout_rate=0.2, activation=tf.nn.relu):
    assert type(input_shape) == int
    assert type(output_shape) == int

    input_ph = tf.placeholder(dtype=tf.float32, shape=[None, input_shape])
    output_ph = tf.placeholder(dtype=tf.float32, shape=[None, output_shape])
    
    policy = Policy(
        num_outputs=output_shape,
        layers=[500, 500],
        dropout_rate=0.2,
        activation=tf.nn.relu)
    output_pred = policy(input_ph)

    mse = tf.reduce_mean(0.5 * tf.square(output_pred - output_ph))
    opt = tf.train.AdamOptimizer().minimize(mse)
    return input_ph, output_ph, output_pred, mse, opt


def train_policy(sess, saver, input_ph, output_ph, mse, opt, observations, actions, batch_size=32, num_steps=10000):
    observations, actions = np.array(observations), np.array(actions)

    print('Training the model')
    # run training
    for training_step in range(num_steps):
        # get a random subset of the training data
        indices = np.random.randint(low=0, high=len(observations), size=batch_size)
        input_batch = observations[indices]
        output_batch = actions[indices]
        
        # run the optimizer and get the mse
        _, mse_run = sess.run([opt, mse], feed_dict={input_ph: input_batch, output_ph: output_batch})
        
        # print the mse every so often
        if training_step % 1000 == 0:
            print('{0:04d} mse: {1:.3f}'.format(training_step, mse_run))
            saver.save(sess, '/tmp/model.ckpt')


def train(args):
    observations, actions = initial_dataset(args.envname)
    actions = actions.reshape((observations.shape[0], -1))
    print('observations', observations.shape, 'actions', actions.shape)
    observations, actions = observations.tolist(), actions.tolist()

    # Get env, expert policy, and imitator policy
    env = gym.make(args.envname)
    expert_policy_fn = load_policy.load_policy('experts/' + args.envname + '.pkl')
    input_ph, output_ph, output_pred, mse, opt = create_imitator_policy()

    with tf.Session() as sess:

        # initialize variables
        sess.run(tf.global_variables_initializer())
        # create saver to save model variables
        saver = tf.train.Saver()

        # # Display the performance before training
        # _, obs_pi, _ = imitator_rollouts(sess, env, input_ph, output_pred,
        #     max_timesteps=args.max_timesteps, num_rollouts=100, render=True)

        for i in range(args.dagger_iters):
            # Train the imitator on the expert policy
            train_policy(
                sess, saver, input_ph, output_ph, mse, opt, observations, 
                actions, batch_size=64, num_steps=30000)

            # Decide when to render progress
            if args.render != 0:
                if (i % args.render) == 0:
                    imitator_rollouts(
                        sess, env, input_ph, output_pred, 
                        max_timesteps=args.max_timesteps, num_rollouts=10, 
                        render=True)

            # Run pi(action_t|theta_t) to get observations_pi = {O_1, ..., O_n}
            _, obs_pi, _ = imitator_rollouts(sess, env, input_ph, output_pred, 
                max_timesteps=args.max_timesteps,
                num_rollouts=args.num_rollouts, render=False)

            # Get the expert labels for observations_pi
            actions_expert = expert_labels(expert_policy_fn, obs_pi)

            # # Aggregate the dataset
            for ob, act in zip(obs_pi, actions_expert):
                observations.append(ob)
                actions.append(act)

        # Display the performance after training
        _, obs_pi, _ = imitator_rollouts(sess, env, input_ph, output_pred,
            max_timesteps=args.max_timesteps, num_rollouts=10, render=True)

    env.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('dagger_iters', type=int)
    parser.add_argument('--render', '-r', type=int, default=5, help='Render every x dagger iterations')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20, help='Number of expert roll outs')
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()