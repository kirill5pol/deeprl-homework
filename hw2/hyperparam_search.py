from multiprocessing import Process
import inspect
import logz
import time
import os

import tensorflow as tf
import numpy as np
import pybulletgym.envs
import pybullet
import gym

from train_pg_f18 import train_PG
import plot


def plotting(logdir, legend=None, value="AverageReturn"):
    use_legend = False
    if legend is not None:
        assert len(legend) == len(
            logdir
        ), "Must give a legend title for each set of experiments."
        use_legend = True

    data = []
    if use_legend:
        for logdir, legend_title in zip(logdir, legend):
            data += plot.get_datasets(logdir, legend_title)
    else:
        for logdir in logdir:
            data += plot.get_datasets(logdir)

    if isinstance(value, list):
        values = value
    else:
        values = [value]
    for value in values:
        plot.plot_data(data, value=value)


def run_training(
    env_name,
    exp_name="vpg",
    render=False,
    discount=1.0,
    n_iter=100,
    batch_size=1000,
    ep_len=-1.0,
    learning_rate=5e-3,
    reward_to_go=True,
    dont_normalize_advantages=False,
    nn_baseline=False,
    seed=1,
    n_experiments=3,
    n_layers=2,
    size=64,
):

    if not (os.path.exists("data")):
        os.makedirs("data")
    logdir = exp_name
    logdir = os.path.join("data", logdir)
    if os.path.exists(logdir):
        raise ValueError("Logging directory already exists!")
    os.makedirs(logdir)

    max_path_length = ep_len if ep_len > 0 else None

    processes = []

    for e in range(n_experiments):
        seed = seed + 10 * e
        print("Running experiment with seed %d" % seed)

        def train_func():
            train_PG(
                exp_name=exp_name,
                env_name=env_name,
                n_iter=n_iter,
                gamma=discount,
                min_timesteps_per_batch=batch_size,
                max_path_length=max_path_length,
                learning_rate=learning_rate,
                reward_to_go=reward_to_go,
                animate=render if e == 0 else False,
                logdir=os.path.join(logdir, "%d" % seed),
                normalize_advantages=not dont_normalize_advantages,
                nn_baseline=nn_baseline,
                seed=seed,
                n_layers=n_layers,
                size=int(size),
            )

        # # Awkward hacky process runs, because Tensorflow does not like
        # # repeatedly calling train_PG in the same thread.
        p = Process(target=train_func, args=tuple())
        p.start()
        processes.append(p)
        # if you comment in the line below, then the loop will block
        # until this process finishes
        # p.join()

    for p in processes:
        p.join()


def grid_search(args):
    discount = 0.9
    n_iter = 100
    # lrs = [0.00001, 0.00003, 0.00006, 0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1]
    # bss = [1000, 2000, 4000, 8000, 16000, 32000, 64000]

    lrs = [0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06]
    bss = [1000, 4000, 16000, 32000]

    all_exps = []
    for lr in lrs:
        for bs in bss:
            exp_name = args.exp_name + " - lr_{} - bs_{}".format(lr, bs)

            run_training(
                args.env_name,
                exp_name=exp_name,
                render=args.render,
                discount=discount,
                n_iter=n_iter,
                #################################
                learning_rate=lr,
                batch_size=bs,
                #################################
                ep_len=-1.0,
                reward_to_go=True,
                dont_normalize_advantages=False,
                nn_baseline=False,
                seed=1,
                n_experiments=3,
                n_layers=2,
                size=64,
            )

            all_exps.append(exp_name)
            # plotting(exp_name) # Plot the current experiment

    plotting(all_exps)  # Plot them all


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type=str)
    parser.add_argument("--exp_name", "-e", type=str, default="vpg")
    parser.add_argument("--render", "-r", action="store_true")

    parser.add_argument("--n_experiments", "-p", type=int, default=3)
    parser.add_argument("--n_layers", "-l", type=int, default=2)
    parser.add_argument("--size", "-s", type=int, default=64)
    args = parser.parse_args()
    grid_search(args)


if __name__ == "__main__":
    main()
