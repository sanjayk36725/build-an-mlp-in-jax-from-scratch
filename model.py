"""
Build an MLP in JAX from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - make_prng_key
import jax.random as random

def make_prng_key(seed):
    return random.PRNGKey(seed)

# Step 2 - split_prng_key
import jax.random as random

def split_prng_key(key, num):
    return random.split(key, num)

# Step 3 - sample_normal_matrix
import jax.random as random

def sample_normal_matrix(key, shape):
    return random.normal(key, shape)

# Step 4 - sample_input_features
import jax.numpy as jnp

def sample_input_features(key, batch_size, num_features):
    return sample_normal_matrix(key, (batch_size, num_features))

# Step 5 - assign_class_labels
import jax.numpy as jnp

def assign_class_labels(inputs, num_classes):
    return jnp.argmax(inputs[:, :num_classes], axis=1).astype(jnp.int32)

# Step 6 - one_hot_encode_labels
def one_hot_encode_labels(labels, num_classes):
    return jnp.eye(num_classes, dtype=jnp.float32)[labels]

# Step 7 - init_linear_layer
import jax.numpy as jnp

def init_linear_layer(key, in_dim, out_dim, scale=0.1):
    return {
        "W": sample_normal_matrix(key, (in_dim, out_dim)) * scale,
        "b": jnp.zeros((out_dim,), dtype=jnp.float32)
    }

# Step 8 - init_mlp_params
def init_mlp_params(key, layer_sizes, scale=0.1):
    keys = split_prng_key(key, len(layer_sizes) - 1)

    return [
        init_linear_layer(keys[i],
                          layer_sizes[i],
                          layer_sizes[i + 1],
                          scale)
        for i in range(len(layer_sizes) - 1)
    ]

# Step 9 - linear_forward
import jax.numpy as jnp

def linear_forward(inputs, layer_params):
    return jnp.dot(inputs, layer_params["W"]) + layer_params["b"]

# Step 10 - relu_activation
import jax.numpy as jnp

def relu_activation(x):
    return jnp.maximum(x, 0)

# Step 11 - softmax_probabilities
import jax.numpy as jnp

def softmax_probabilities(logits):
    shifted = logits - jnp.max(logits, axis=-1, keepdims=True)
    exp_logits = jnp.exp(shifted)
    return exp_logits / jnp.sum(exp_logits, axis=-1, keepdims=True)

# Step 12 - mlp_forward (not yet solved)
# TODO: implement

# Step 13 - log_softmax_logits (not yet solved)
# TODO: implement

# Step 14 - cross_entropy_loss (not yet solved)
# TODO: implement

# Step 15 - classification_accuracy (not yet solved)
# TODO: implement

# Step 16 - loss_fn_of_params (not yet solved)
# TODO: implement

# Step 17 - compute_param_grads (not yet solved)
# TODO: implement

# Step 18 - sgd_update_params (not yet solved)
# TODO: implement

# Step 19 - training_step (not yet solved)
# TODO: implement

# Step 20 - train_mlp (not yet solved)
# TODO: implement

# Step 21 - predict_classes (not yet solved)
# TODO: implement

