import tensorflow as tf

# Adaptive Threshold Updates
tau = tf.Variable([0.7, 0.7], dtype=tf.float32, name="tau")
gamma = tf.Variable([0.2, 0.2], dtype=tf.float32, name="gamma")
class_frequencies = tf.Variable([0.5, 0.5], dtype=tf.float32, name="class_frequencies")

alpha, eta, beta, zeta = 0.01, 0.001, 0.01, 0.001
m, n = 2, 2
eps = 1e-6

# Monte Carlo Dropout
def mc_dropout_forward_passes(model, x1, x2, T):
    """Perform T stochastic forward passes"""
    mc_preds = []
    for _ in range(T):
        pred = model([x1, x2], training=True)
        mc_preds.append(pred)
    return tf.stack(mc_preds, axis=0)

def compute_confidence_variance(mc_preds):
    """Compute predictive mean and variance"""
    mean_probs = tf.reduce_mean(mc_preds, axis=0)
    var_probs = tf.math.reduce_variance(mc_preds, axis=0)
    return mean_probs, var_probs

def update_confidence_threshold(mean_pred_conf, class_freq):
    """Update Confidence threshold"""
    log_term = tf.math.log(1.0 / (1.0 - class_freq + eps))
    imbalance_term = eta * tf.pow(class_freq, m) * log_term
    delta_tau = alpha * tf.nn.relu(mean_pred_conf - tau) + imbalance_term
    tau.assign_add(delta_tau)
    tau.assign(tf.clip_by_value(tau, 0.7, 0.97))

def update_uncertainty_threshold(mean_pred_var, class_freq):
    """Update uncertainty threshold"""
    log_term = tf.math.log(1.0 / (1.0 - class_freq + eps))
    imbalance_term = zeta * tf.pow(class_freq, n) * log_term
    delta_gamma = - beta * tf.nn.relu(gamma - mean_pred_var) - imbalance_term
    gamma.assign_add(delta_gamma)
    gamma.assign(tf.clip_by_value(gamma, 0.01, 0.2))


# Pseudo-label Filtering

def get_pseudo_mask(mean_conf, mean_var):
    """Filter pseudo labels based on τ and γ"""
    valid_mask = (mean_conf > tau) & (mean_var < gamma)
    masked_conf = tf.where(valid_mask, mean_conf, -1e9)
    hard_preds = tf.argmax(masked_conf, axis=-1)
    has_valid = tf.reduce_any(valid_mask, axis=-1)
    pseudo_labels = tf.where(has_valid, hard_preds, -tf.ones_like(hard_preds, dtype=tf.int64))
    return tf.cast(tf.expand_dims(pseudo_labels, axis=-1), tf.float32)

# Loss Wrapper
def adaptive_weighted_loss(y_true, y_pred, pseudo_mask, weights, loss_fn):
    """Apply adaptive weights for class imbalance"""
    weights = weights / (tf.reduce_sum(weights) + eps)
    loss_per_pixel = loss_fn(y_true, y_pred)
    y_class = tf.cast(tf.round(y_true), tf.int32)
    mask_class0 = pseudo_mask * tf.cast(tf.equal(y_class, 0), tf.float32)
    mask_class1 = pseudo_mask * tf.cast(tf.equal(y_class, 1), tf.float32)
    weighted_loss_class0 = weights[0] * mask_class0 * loss_per_pixel
    weighted_loss_class1 = weights[1] * mask_class1 * loss_per_pixel
    total_loss = tf.reduce_sum(weighted_loss_class0 + weighted_loss_class1)
    num_confident_pixels = tf.reduce_sum(pseudo_mask) + eps

    return total_loss / num_confident_pixels
