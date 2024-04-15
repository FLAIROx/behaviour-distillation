import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
import gymnax
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper
import time

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

        return actor_mean


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config):
    """Create training function based on config."""
    config["NUM_UPDATES"] = config["UPDATE_EPOCHS"]

    env, env_params = gymnax.make(config["ENV_NAME"])
    env_params = env_params.replace(**config["ENV_PARAMS"])
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    # Do I need a schedule on the LR for BC?
    def linear_schedule(count):
        frac = 1.0 - (count // config["NUM_UPDATES"])
        return config["LR"] * frac

    def train(synth_data, action_labels, rng):
        """Train using BC on synthetic data with fixed action labels and evaluate on RL environment"""

        # 1. INIT NETWORK AND TRAIN STATE
        # TODO: This is hacky. Fix it to handle continuous actions elegantly

        if "Continuous" in config["ENV_NAME"] or "Brax" in config["ENV_NAME"]:
            action_shape = env.action_space().shape[0]
            is_continuous = True
        else:
            action_shape = env.action_space().n
            is_continuous = False
        network = BCAgent(
            action_shape, activation=config["ACTIVATION"], width=config["WIDTH"]
        )

        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)

        assert (
                synth_data[0].shape == env.observation_space(env_params).shape
        ), f"Data of shape {synth_data[0].shape} does not match env observations of shape {env.observation_space(env_params).shape}"

        # Setup optimizer
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )

        # Train state carries everything needed for NN training
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # 2. BC TRAIN LOOP
        def _bc_train(train_state, rng):
            def _bc_update_step(bc_state, unused):
                train_state, rng = bc_state

                def _loss_and_acc(params, apply_fn, step_data, y_true, num_classes):
                    """Compute cross-entropy loss and accuracy."""
                    y_pred = apply_fn(params, step_data)
                    acc = jnp.mean(jnp.argmax(y_pred, axis=-1) == y_true)
                    labels = jax.nn.one_hot(y_true, num_classes)
                    loss = -jnp.sum(labels * jax.nn.log_softmax(y_pred))
                    loss /= labels.shape[0]
                    return loss, acc

                # TODO: Add loss function for continuous case --> XENT?
                grad_fn = jax.value_and_grad(_loss_and_acc, has_aux=True)

                # Not needed if using entire dataset
                rng, perm_rng = jax.random.split(rng)
                perm = jax.random.permutation(perm_rng, len(action_labels))
                step_data = synth_data[perm]
                y_true = action_labels[perm]

                loss_and_acc, grads = grad_fn(
                    train_state.params,
                    train_state.apply_fn,
                    step_data,
                    y_true,
                    action_shape,
                )
                train_state = train_state.apply_gradients(grads=grads)
                bc_state = (train_state, rng)
                return bc_state, loss_and_acc

            bc_state = (train_state, rng)
            bc_state, loss_and_acc = jax.lax.scan(
                _bc_update_step, bc_state, None, config["UPDATE_EPOCHS"]
            )
            loss, acc = loss_and_acc
            return bc_state, loss, acc

        rng, _rng = jax.random.split(rng)
        bc_state, bc_loss, bc_acc = _bc_train(train_state, _rng)
        train_state = bc_state[0]

        # Init envs
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        # 3. POLICY EVAL LOOP
        def _eval_ep(runner_state):
            # Environment stepper
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # Select Action
                rng, _rng = jax.random.split(rng)
                pi = train_state.apply_fn(train_state.params, last_obs)
                if config["GREEDY_ACT"]:
                    action = pi.argmax(
                        axis=-1
                    )  # if 2+ actions are equiprobable, returns first
                else:
                    probs = distrax.Categorical(logits=pi)
                    action = probs.sample(seed=_rng)

                # Step env
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)
                transition = Transition(
                    done, action, -1, reward, jax.nn.log_softmax(pi), last_obs, info
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            metric = traj_batch.info
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = _eval_ep(runner_state)

        metric["bc_loss"] = bc_loss
        metric["bc_accuracy"] = bc_acc

        return {"runner_state": runner_state, "metrics": metric}

    return train


