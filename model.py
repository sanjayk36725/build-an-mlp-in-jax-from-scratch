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

# Step 12 - mlp_forward
def mlp_forward(params, x):
    for layer in params[:-1]:
        x = relu_activation(linear_forward(x, layer))

    return linear_forward(x, params[-1])

# Step 13 - log_softmax_logits
import jax.numpy as jnp

def log_softmax_logits(logits):
    shifted = logits - jnp.max(logits, axis=-1, keepdims=True)
    logsumexp = jnp.log(jnp.sum(jnp.exp(shifted), axis=-1, keepdims=True))
    return shifted - logsumexp

# Step 14 - cross_entropy_loss
import jax.numpy as jnp

def cross_entropy_loss(logits, one_hot_targets):
    log_probs = log_softmax_logits(logits)
    return -jnp.mean(jnp.sum(one_hot_targets * log_probs, axis=-1))

# Step 15 - classification_accuracy
import jax.numpy as jnp

def classification_accuracy(logits, labels):
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == labels)

# Step 16 - loss_fn_of_params
def loss_fn_of_params(params, inputs, one_hot_targets):
    logits = mlp_forward(params, inputs)
    return cross_entropy_loss(logits, one_hot_targets)

# Step 17 - compute_param_grads
import jax

def compute_param_grads(params, x, one_hot_targets):
    return jax.grad(loss_fn_of_params)(params, x, one_hot_targets)

# Step 18 - sgd_update_params
def sgd_update_params(params, grads, lr):
    return [
        {
            "W": p["W"] - lr * g["W"],
            "b": p["b"] - lr * g["b"]
        }
        for p, g in zip(params, grads)
    ]

# Step 19 - training_step
def training_step(params, x, one_hot_targets, learning_rate):
    loss = loss_fn_of_params(params, x, one_hot_targets)
    grads = compute_param_grads(params, x, one_hot_targets)
    new_params = sgd_update_params(params, grads, learning_rate)
    return new_params, loss

# Step 20 - train_mlp
def train_mlp(params, x, one_hot_targets, learning_rate, num_epochs):
    for _ in range(num_epochs):
        params, _ = training_step(
            params,
            x,
            one_hot_targets,
            learning_rate
        )
    return params

# Step 21 - predict_classes
import jax.numpy as jnp

def predict_classes(params, x):
    logits = mlp_forward(params, x)
    return jnp.argmax(logits, axis=-1).astype(jnp.int32)

