import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
from flax import struct
import distrax
import gymnax
from gymnax.environments import environment, spaces
from gymnax.wrappers.purerl import GymnaxWrapper
import brax
from brax import envs
from brax.envs.wrappers.training import EpisodeWrapper, AutoResetWrapper
# from brax.envs.wrapper import EpisodeWrapper, AutoResetWrapper
from evosax import OpenES, ParameterReshaper

import wandb

import sys
sys.path.insert(0, '..')
# sys.path.insert(0, '/home/clu/explainable-policies')
from purejaxrl.wrappers import (
    LogWrapper,
    BraxGymnaxWrapper,
    VecEnv,
    NormalizeVecObservation,
    NormalizeVecReward,
    ClipAction,
    TransformObservation,
)
import time
import argparse
import pickle as pkl
import os


def wrap_brax_env(env, normalize_obs=True, normalize_reward=True, gamma=0.99):
    """Apply standard set of Brax wrappers"""
    env = LogWrapper(env)
    env = ClipAction(env)
    env = VecEnv(env)
    if normalize_obs:
        env = NormalizeVecObservation(env)
    if normalize_reward:
        env = NormalizeVecReward(env, gamma)
    return env

# Continuous action BC agent
class BCAgentContinuous(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"
    width: int = 64 #512 for Brax

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            self.width, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.width, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        return pi


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config):
    """Create training function based on config. The returned function will:
    - Train a policy through BC
    - Evaluate the policy in the environment
    """

    env, env_params = BraxGymnaxWrapper(config["ENV_NAME"]), None
    env = LogWrapper(env)
    env = ClipAction(env)
    if config["CONST_NORMALIZE_OBS"]:
        func = lambda x: (x - config["OBS_MEAN"]) / jnp.sqrt(config["OBS_VAR"] + 1e-8)
        env = TransformObservation(env, func)
    env = VecEnv(env)
    if config["NORMALIZE_OBS"] and not config["CONST_NORMALIZE_OBS"]:
        env = NormalizeVecObservation(env)
    if config["NORMALIZE_REWARD"]:
        env = NormalizeVecReward(env, config["GAMMA"])

    def train(params, rng):

        action_shape = env.action_space(env_params).shape[0]
        network = BCAgentContinuous(
            action_shape, activation=config["ACTIVATION"], width=config["WIDTH"]
        )

        # Init envs
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        #         obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        obsv, env_state = env.reset(reset_rng, env_params)

        # 3. POLICY EVAL LOOP
        def _eval_ep(runner_state):
            # Environment stepper
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # Select Action
                rng, _rng = jax.random.split(rng)
                pi = network.apply(params, last_obs)
                if config["GREEDY_ACT"]:
                    action = pi.argmax(
                        axis=-1
                    )  # if 2+ actions are equiprobable, returns first
                else:
                    action = pi.sample(seed=_rng)

                # Step env
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                obsv, env_state, reward, done, info = env.step(rng_step, env_state, action, env_params)
                transition = Transition(
                    done, action, -1, reward, pi.log_prob(action), last_obs, info
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            metric = traj_batch.info
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (params, env_state, obsv, _rng)
        runner_state, metric = _eval_ep(runner_state)

        return {"runner_state": runner_state, "metrics": metric}

    return train


def init_env(config):
    """Initialize environment"""
    env, env_params = BraxGymnaxWrapper(config["ENV_NAME"]), None
    env = wrap_brax_env(env, normalize_obs=config["NORMALIZE_OBS"], normalize_reward=config["NORMALIZE_REWARD"])
    return env, env_params


def init_params(env, env_params, config):
    """Initialize dataset to be learned"""
    action_shape = env.action_space(env_params).shape[0]
    network = BCAgentContinuous(
        action_shape, activation=config["ACTIVATION"], width=config["WIDTH"]
    )

    rng = jax.random.PRNGKey(0)
    init_x = jnp.zeros((1, *env.observation_space(env_params).shape))
    params = network.init(rng, init_x)

    param_reshaper = ParameterReshaper(params)
    return params, param_reshaper 


def init_es(rng_init, param_reshaper, es_config):
    """Initialize OpenES strategy"""
    strategy = OpenES(
        popsize=es_config["popsize"],
        num_dims=param_reshaper.total_params,
        opt_name="adam",
        maximize=True,
    )
    # Replace state mean with real observations
    # state = state.replace(mean = sampled_data)

    es_params = strategy.default_params
    es_params = es_params.replace(sigma_init=es_config["sigma_init"], sigma_decay=es_config["sigma_decay"])
    state = strategy.initialize(rng_init, es_params)

    return strategy, es_params, state


def parse_arguments(argstring=None):
    """Parse arguments either from `argstring` if not None or from command line otherwise"""
    parser = argparse.ArgumentParser()
    # Default arguments should result in ~1600 return in Hopper

    # Outer loop args
    parser.add_argument(
        "--env",
        type=str,
        help="Brax environment name",
        default="hopper"
    )
    parser.add_argument(
        "--popsize",
        type=int,
        help="ES population size",
        default=2048
    )
    parser.add_argument(
        "--generations",
        type=int,
        help="Number of ES generations",
        default=2000
    )
    parser.add_argument(
        "--rollouts",
        type=int,
        help="Number of BC policies trained per candidate",
        default=1
    )
    parser.add_argument(
        "--sigma_init",
        type=float,
        help="Initial ES variance",
        default=0.03
    )
    parser.add_argument(
        "--sigma_decay",
        type=float,
        help="ES variance decay factor",
        default=1.0
    )

    # Inner loop args
    parser.add_argument(
        "--eval_envs",
        type=int,
        help="Number of evaluation environments",
        default=4
    )
    parser.add_argument(
        "--activation",
        type=str,
        help="NN nonlinearlity type (relu/tanh)",
        default="tanh"
    )
    parser.add_argument(
        "--width",
        type=int,
        help="NN width",
        default=512
    )
    parser.add_argument(
        "--const_normalize_obs",
        type=int,
        default=0
    )
    parser.add_argument(
        "--normalize_obs",
        type=int,
        default=1
    )
    parser.add_argument(
        "--normalize_reward",
        type=int,
        default=1
    )

    # Misc. args
    parser.add_argument(
        "--seed",
        type=int,
        help="RNG seed",
        default=1337
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        help="Num. generations between logs",
        default=1
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        help="Num. generations between data saves",
        default=10
    )
    parser.add_argument(
        "--folder",
        type=str,
        help="Path to save folder",
        default="../results/"
    )
    parser.add_argument(
        "--project",
        type=str,
        help="wandb Project Name",
        default="ES-Baselines-ICLR"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False
    )
    if argstring is not None:
        args = parser.parse_args(argstring.split())
    else:
        args = parser.parse_args()

    if args.folder[-1] != "/":
        args.folder = args.folder + "/"

    return args


def make_configs(args):
    config = {
        "NUM_ENVS": args.eval_envs,  # 8 # Num eval envs for each BC policy
        "NUM_STEPS": 1024,  # 128 # Max num eval steps per env
        "ACTIVATION": args.activation,
        "WIDTH": args.width,
        "ENV_NAME": args.env,
        "GREEDY_ACT": False,  # Whether to use greedy act in env or sample
        "ENV_PARAMS": {},
        "GAMMA": 0.99,
        "CONST_NORMALIZE_OBS": bool(args.const_normalize_obs),
        "NORMALIZE_OBS": bool(args.normalize_obs),
        "NORMALIZE_REWARD": bool(args.normalize_reward),
        "DEBUG": args.debug,
        "SEED": args.seed,
        "FOLDER": args.folder,
        "PROJECT": args.project
    }
    es_config = {
        "popsize": args.popsize,  # Num of candidates (variations) generated every generation
        "rollouts_per_candidate": args.rollouts,  # 32 Num of BC policies trained per candidate
        "n_generations": args.generations,
        "log_interval": args.log_interval,
        "sigma_init": args.sigma_init,
        "sigma_decay": args.sigma_decay,
        "save_interval": args.save_interval
    }
    return config, es_config


def main(config, es_config):

    print("config")
    print("-----------------------------")
    for k, v in config.items():
        print(f"{k} : {v},")
    print("-----------------------------")
    print("ES_CONFIG")
    for k, v in es_config.items():
        print(f"{k} : {v},")

    # Setup wandb
    if not config["DEBUG"]:
        wandb_config = config.copy()
        wandb_config["es_config"] = es_config
        wandb_run = wandb.init(project=config["PROJECT"], config=wandb_config)

    # Load here so that OBS_MEAN and OBS_VAR are not logged to wandb, since they are massive arrays
    if config["CONST_NORMALIZE_OBS"]:
        config["OBS_MEAN"] = jnp.load(f"../normalize_params/mean_{config['ENV_NAME']}.npy")
        config["OBS_VAR"] = jnp.load(f"../normalize_params/var_{config['ENV_NAME']}.npy")

    # Init environment and dataset (params)
    env, env_params = init_env(config)
    _, param_reshaper = init_params(env, env_params, config)

    rng = jax.random.PRNGKey(config["SEED"])

    # Initialize OpenES Strategy
    rng, rng_init = jax.random.split(rng)
    strategy, es_params, state = init_es(rng_init, param_reshaper, es_config)

    # Set up vectorized fitness function
    train_fn = make_train(config)

    def single_seed_eval(rng_input, params):
        out = train_fn(params, rng_input)
        return out

    multi_seed_eval = jax.vmap(single_seed_eval, in_axes=(0, None))  # Vectorize over seeds
    train_and_eval = jax.jit(jax.vmap(multi_seed_eval, in_axes=(None, 0)))  # Vectorize over datasets

    if len(jax.devices()) > 1:
        # If available, distribute over multiple GPUs
        train_and_eval = jax.pmap(train_and_eval, in_axes=(None, 0))

    start = time.time()
    lap_start = start
    fitness_over_gen = []
    max_fitness_over_gen = []
    for gen in range(es_config["n_generations"]):
        # Gen new dataset
        rng, rng_ask, rng_inner = jax.random.split(rng, 3)
        members, state = jax.jit(strategy.ask)(rng_ask, state, es_params)
        # Eval fitness
        batch_rng = jax.random.split(rng_inner, es_config["rollouts_per_candidate"])
        # Preemptively overwrite to reduce memory load
        out = None
        returns = None
        dones = None
        fitness = None
        shaped_datasets = None

        with jax.disable_jit(config["DEBUG"]):
            params = param_reshaper.reshape(members)

            out = train_and_eval(batch_rng, params)

            returns = out["metrics"]["returned_episode_returns"]  # dim=(popsize, rollouts, num_steps, num_envs)
            ep_lengths = out["metrics"]["returned_episode_lengths"]
            dones = out["metrics"]["returned_episode"]  # same dim, True for last steps, False otherwise

            mean_ep_length = (ep_lengths * dones).sum(axis=(-1, -2, -3)) / dones.sum(
                axis=(-1, -2, -3))
            mean_ep_length = mean_ep_length.flatten()

            # Division by zero, watch out
            fitness = (returns * dones).sum(axis=(-1, -2, -3)) / dones.sum(
                axis=(-1, -2, -3))  # fitness, dim = (popsize)
            # fitness = out["metrics"]["returned_episode_returns"][:, :, -1, :].mean(axis=(-1, -2))
            fitness = fitness.flatten()  # Necessary if pmap-ing to 2+ devices

        # Update ES strategy with fitness info
        state = jax.jit(strategy.tell)(members, fitness, state, es_params)
        fitness_over_gen.append(fitness.mean())
        max_fitness_over_gen.append(fitness.max())

        # Logging
        if gen % es_config["log_interval"] == 0 or gen == 0:
            lap_end = time.time()

            print(
                f"Gen: {gen}, Fitness: {fitness.mean():.2f} +/- {fitness.std():.2f}, "
                + f"Best: {state.best_fitness:.2f}, Lap time: {lap_end - lap_start:.1f}s"
            )
            if not config["DEBUG"]:
                wandb.log({
                    f"{config['ENV_NAME']}:mean_fitness": fitness.mean(),
                    f"{config['ENV_NAME']}:fitness_std": fitness.std(),
                    f"{config['ENV_NAME']}:max_fitness": fitness.max(),
                    "mean_ep_length": mean_ep_length.mean(),
                    "max_ep_length": mean_ep_length.max(),
                    "mean_fitness": fitness.mean(),
                    "max_fitness": fitness.max(),
                    "Gen time": lap_end - lap_start,
                })
            lap_start = lap_end

        if gen % es_config["save_interval"] == 0 or gen == 0:
            data = {
                "state": state,
                "fitness_over_gen": fitness_over_gen,
                "max_fitness_over_gen": max_fitness_over_gen,
                "fitness": fitness,
                "config": config,
                "es_config": es_config
            }

            directory = config["FOLDER"]
            if not os.path.exists(directory):
                os.mkdir(directory)
            filename = directory + f"data_gen{gen}.pkl"
            file = open(filename, 'wb')
            pkl.dump(data, file)
            file.close()

    print(f"Total time: {(lap_end - start) / 60:.1f}min")

    data = {
        "state": state,
        "fitness_over_gen": fitness_over_gen,
        "max_fitness_over_gen": max_fitness_over_gen,
        "fitness": fitness,
        "config": config,
        "es_config": es_config
    }

    directory = config["FOLDER"]
    if not os.path.exists(directory):
        os.mkdir(directory)
    filename = directory + "data_final.pkl"
    file = open(filename, 'wb')
    pkl.dump(data, file)
    file.close()


def train_from_arg_string(argstring):
    """Launches training from an argument string of the form
    `--env humanoid --popsize 1024 --epochs 200 ...`
    Main use case is in conjunction with Submitit for creating job arrays
    """
    args = parse_arguments(argstring)
    config, es_config = make_configs(args)
    main(config, es_config)


if __name__ == "__main__":
    args = parse_arguments()
    config, es_config = make_configs(args)
    main(config, es_config)
