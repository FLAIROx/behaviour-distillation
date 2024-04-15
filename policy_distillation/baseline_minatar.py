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
from evosax import OpenES, ParameterReshaper, SNES
from typing import Any, Optional, Union, Tuple
Array = Any
import chex

import wandb

import sys
sys.path.insert(0, '..')
from purejaxrl.wrappers import (
    LogWrapper,
    VecEnv,
    FlattenObservationWrapper,
    NormalizeVecObservation,
    NormalizeVecReward,
    TransformObservation,
)
import time
import argparse
import pickle as pkl
import os


def wrap_minatar_env(env, config, normalize_obs=False, normalize_reward=False, gamma=0.99):
    """Apply standard set of Brax wrappers"""
    if config["NET"].lower() == "mlp":
        env = FlattenObservationWrapper(env)
        env = LogWrapper(env)
        env = VecEnv(env)
        if normalize_obs:
            env = NormalizeVecObservation(env)
        if normalize_reward:
            env = NormalizeVecReward(env, gamma)
    else:
        print("This is an implementation shortcut.")
    return env


class BCAgent(nn.Module):
    """Network architecture. Matches MinAtar PPO agent from PureJaxRL"""

    action_dim: Sequence[int]
    activation: str = "tanh"
    width: int = 64

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
        pi = distrax.Categorical(logits=actor_mean)

        return pi

def default_mlp_init(scale=0.05):
    return nn.initializers.uniform(scale)


class MinAtarCNN(nn.Module):
    """A general purpose conv net model."""

    action_dim: int
    activation: str = "relu"
    hidden_dims: Sequence[int] = 64

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        x = nn.Conv(
            features=16,
            kernel_size=(3, 3),
            padding="SAME",
            strides=1,
            bias_init=default_mlp_init(),
        )(x)
        x = activation(x)
        x = x.reshape((x.shape[0], -1))
        for hidden_dim in self.hidden_dims:
            x = activation(
                nn.Dense(
                    features=hidden_dim,
                    bias_init=default_mlp_init(),
                )(x)
            )
        x = nn.Dense(features=self.action_dim)(x)
        pi = distrax.Categorical(logits=x)
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

    env, env_params = gymnax.make(config["ENV_NAME"])
    if config["NET"].lower() == "mlp":
        env = FlattenObservationWrapper(env)
        env = LogWrapper(env)
        env = VecEnv(env)
        if config["CONST_NORMALIZE_OBS"]:
            func = lambda x: (x - config["OBS_MEAN"]) / jnp.sqrt(config["OBS_VAR"] + 1e-8)
            env = TransformObservation(env, func)
    else:
        config["OBS_MEAN"] = config["OBS_MEAN"].reshape((10, 10, -1))
        config["OBS_VAR"] = config["OBS_VAR"].reshape((10, 10, -1))
        env = LogWrapper(env)
        env = VecEnv(env)
        if config["CONST_NORMALIZE_OBS"]:
            func = lambda x: (x - config["OBS_MEAN"]) / jnp.sqrt(config["OBS_VAR"] + 1e-8)
            env = TransformObservation(env, func)

    def train(params, rng):

        if config["NET"].lower() == "mlp":
            network = BCAgent(
                env.action_space(env_params).n, activation=config["ACTIVATION"], width=config["WIDTH"]
            )
        elif config["NET"].lower() == "cnn":
            network = MinAtarCNN(
                env.action_space(env_params).n, activation=config["ACTIVATION"], hidden_dims=[config["WIDTH"]]*config["FFWD_LAYERS"]
            )

        # Init envs
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = env.reset(reset_rng, env_params)

        # 1. POLICY EVAL LOOP
        def _eval_ep(runner_state):
            # Environment stepper
            def _env_step(runner_state, unused):
                params, env_state, last_obs, rng = runner_state

                # Select Action
                rng, _rng = jax.random.split(rng)
                pi = network.apply(params, last_obs)
                if config["GREEDY_ACT"]:
                    action = pi.probs.argmax(
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
                runner_state = (params, env_state, obsv, rng)
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
    env, env_params = gymnax.make(config["ENV_NAME"])
    env = wrap_minatar_env(env, config)
    return env, env_params


def init_params(env, env_params, config):
    """Initialize params to be learned"""
    if config["NET"].lower() == "mlp":
        network = BCAgent(
            env.action_space(env_params).n, activation=config["ACTIVATION"], width=config["WIDTH"]
        )
    elif config["NET"].lower() == "cnn":
        network = MinAtarCNN(
            env.action_space(env_params).n, activation=config["ACTIVATION"], hidden_dims=[config["WIDTH"]]*config["FFWD_LAYERS"]
        )

    rng = jax.random.PRNGKey(0)
    init_x = jnp.zeros((1, *env.observation_space(env_params).shape))
    params = network.init(rng, init_x)

    param_reshaper = ParameterReshaper(params)
    return params, param_reshaper 


def init_es(rng_init, param_reshaper, es_config):
    """Initialize OpenES strategy"""
    if es_config["strategy"] == "OpenES":
        strategy = OpenES(
            popsize=es_config["popsize"],
            num_dims=param_reshaper.total_params,
            opt_name="adam",
            maximize=True,         # Maximize=False because the fitness is the train loss
            lrate_init=es_config["lrate_init"],  # Passing it here since for some reason cannot update it in params.replace
            lrate_decay=es_config["lrate_decay"]
        )

        es_params = strategy.params_strategy
        es_params = es_params.replace(sigma_init=es_config["sigma_init"], sigma_limit=es_config["sigma_limit"], sigma_decay=es_config["sigma_decay"])
        state = strategy.initialize(rng_init, es_params)
    elif es_config["strategy"] == "SNES":
        strategy = SNES(
            popsize=es_config["popsize"],
            num_dims=param_reshaper.total_params,
            maximize=True,         # Maximize=False because the fitness is the train loss
        )

        es_params = strategy.params_strategy
        es_params = es_params.replace(sigma_init=es_config["sigma_init"], temperature=es_config["temperature"])
        state = strategy.initialize(rng_init, es_params)
    else:
        raise NotImplementedError

    return strategy, es_params, state


def parse_arguments(argstring=None):
    """Parse arguments either from `argstring` if not None or from command line otherwise"""
    parser = argparse.ArgumentParser()
    # Default arguments should result in ~1600 return in Hopper

    # Outer loop args
    parser.add_argument(
        "--env",
        type=str,
        help="Gymnax environment name: Pong-misc, [Breakout/SpaceInvaders/Freeway/Asterix]-MinAtar",
        default="SpaceInvaders-MinAtar"
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
        "--sigma_limit",
        type=float,
        help="ES variance lower bound (if decaying)",
        default=0.01
    )
    parser.add_argument(
        "--sigma_decay",
        type=float,
        help="ES variance decay factor",
        default=1.0
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="SNES temperature",
        default=20.0
    )
    parser.add_argument(
        "--lrate_init",
        type=float,
        help="ES initial lrate",
        default=0.05
    )
    parser.add_argument(
        "--lrate_decay",
        type=float,
        help="ES lrate decay factor",
        default=1.0
    )
    parser.add_argument(
        "--es_strategy",
        type=str,
        help="Type of es strategy. Have OpenES and SNES",
        default="SNES",
    )

    # Inner loop args
    parser.add_argument(
        "--net",
        type=str,
        help="MLP / CNN",
        default="mlp"
    )
    parser.add_argument(
        "--eval_envs",
        type=int,
        help="Number of evaluation environments",
        default=8
    )
    parser.add_argument(
        "--activation",
        type=str,
        help="NN nonlinearlity type (relu/tanh)",
        default="relu"
    )
    parser.add_argument(
        "--width",
        type=int,
        help="NN width",
        default=512
    )
    parser.add_argument(
        "--ffwd_layers",
        type=int,
        help="CNN number of ffwd layers",
        default=2
    )
    parser.add_argument(
        "--const_normalize_obs",
        type=int,
        default=1
    )
    parser.add_argument(
        "--normalize_obs",
        type=int,
        default=0
    )
    parser.add_argument(
        "--normalize_reward",
        type=int,
        default=0
    )
    parser.add_argument(
        "--greedy_act",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=1024,
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
        "NET": args.net,
        "NUM_ENVS": args.eval_envs,  # 8 # Num eval envs for each BC policy
        "NUM_STEPS": args.num_steps,  # 128 # Max num eval steps per env
        "ACTIVATION": args.activation,
        "WIDTH": args.width,
        "FFWD_LAYERS": args.ffwd_layers,
        "ENV_NAME": args.env,
        "GREEDY_ACT": args.greedy_act,  # Whether to use greedy act in env or sample
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
        "sigma_limit": args.sigma_limit,
        "sigma_decay": args.sigma_decay,
        "lrate_init": args.lrate_init,
        "lrate_decay": args.lrate_decay,
        "strategy": args.es_strategy,
        "temperature": args.temperature,
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
    # vmap over images only
    train_and_eval = jax.jit(jax.vmap(multi_seed_eval, in_axes=(None, 0)))  # Vectorize over datasets
        
    if not config["DEBUG"]:
        train_and_eval = jax.jit(train_and_eval)

    # TODO: Refactor to allow for different RNGs for each dataset
    if len(jax.devices()) > 1:

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

        with jax.disable_jit(config["DEBUG"]):
            params = param_reshaper.reshape(members)

            out = train_and_eval(batch_rng, params)

            returns = out["metrics"]["returned_episode_returns"]  # dim=(popsize, rollouts, num_steps, num_envs)
            ep_lengths = out["metrics"]["returned_episode_lengths"]
            dones = out["metrics"]["returned_episode"]  # same dim, True for last steps, False otherwise

            mean_ep_length = (ep_lengths * dones).sum(axis=(-1, -2, -3)) / dones.sum(
                axis=(-1, -2, -3))
            mean_ep_length = mean_ep_length.flatten()

            fitness = out["metrics"]["returned_episode_returns"][:, :, -1, :].mean(axis=(-1, -2))
            fitness = fitness.flatten()  # Necessary if pmap-ing to 2+ devices
        #         fitness = jnp.minimum(fitness, fitness.mean()+40)

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

