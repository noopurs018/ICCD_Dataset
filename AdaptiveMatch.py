# AdaptiveMatch.py
import tensorflow as tf
import numpy as np

class AdaptiveMatch:
    """
    Adaptive Match for Semi-Supervised Learning
    Implements adaptive confidence and uncertainty thresholds with class balancing
    Based on the paper: "Towards Scalable Urban Monitoring: Semi-Supervised Building Change Detection and the ICCD Dataset for the Indian Cities"
    """
    
    def __init__(self, num_classes=2, lambda_decay=0.9, eps=1e-6):
        """
        Initialize adaptive thresholds and hyperparameters
        
        Args:
            num_classes: Number of classes (default: 2 for binary change detection)
            lambda_decay: EMA decay factor for class frequency updates
            eps: Numerical stability constant
        """
        # Adaptive thresholds as trainable variables
        self.tau = tf.Variable([0.7] * num_classes, dtype=tf.float32, name="tau")  # Confidence threshold
        self.gamma = tf.Variable([0.2] * num_classes, dtype=tf.float32, name="gamma")  # Uncertainty threshold
        self.class_frequencies = tf.Variable([1.0/num_classes] * num_classes, 
                                           dtype=tf.float32, name="class_frequencies")
        
        # Hyperparameters from paper
        self.alpha = 0.01  # τ update speed 
        self.eta = 0.001    # τ class balance strength 
        self.beta = 0.01   # γ update speed 
        self.zeta = 0.001   # γ class balance strength 
        self.m = 2          # τ frequency exponent 
        self.n = 2          # γ frequency exponent 
        self.lambda_decay = lambda_decay  # EMA decay for class frequencies
        self.eps = eps      # Numerical stability
        self.num_classes = num_classes
    
    def mc_dropout_forward_passes(self, model, x1, x2, T=15):
        """
        Perform T Monte Carlo dropout forward passes for uncertainty estimation
        
        Args:
            model: Model with dropout layers
            x1, x2: Input image pairs
            T: Number of Monte Carlo samples
            
        Returns:
            Stacked predictions [T, B, H, W, C]
        """
        mc_preds = []
        for _ in range(T):
            pred = model([x1, x2], training=True)  # training=True enables dropout
            mc_preds.append(pred)
        return tf.stack(mc_preds, axis=0)
    
    def compute_confidence_variance(self, mc_preds):
        """
        Compute mean confidence and variance from Monte Carlo predictions
        
        Args:
            mc_preds: [T, B, H, W, C] stacked predictions
            
        Returns:
            mean_probs: [B, H, W, C] mean predictions
            var_probs: [B, H, W, C] variance of predictions
        """
        mean_probs = tf.reduce_mean(mc_preds, axis=0)  # Mean confidence
        var_probs = tf.math.reduce_variance(mc_preds, axis=0)  # Uncertainty
        return mean_probs, var_probs
    
    def compute_classwise_means(self, mean_pred, mean_var):
        """
        Compute class-wise mean confidence and variance
        
        Args:
            mean_pred: [B, H, W, C] mean predictions
            mean_var: [B, H, W, C] variance of predictions
            
        Returns:
            mean_conf: [C] mean confidence per class
            mean_var: [C] mean variance per class
        """
        mean_conf_list = []
        mean_var_list = []
        
        for c in range(self.num_classes):
            # Extract predictions for class c
            p_c = mean_pred[..., c]  # [B, H, W]
            var_c = mean_var[..., c]  # [B, H, W]
            
            # Determine confident pixels for this class
            if c == 0:  # Background class
                mask = tf.cast(p_c < 0.5, tf.float32)  # Confidently background
                conf = (1.0 - p_c) * mask  # Confidence for background
            else:  # Change class
                mask = tf.cast(p_c >= 0.5, tf.float32)  # Confidently change
                conf = p_c * mask  # Confidence for change
            
            # Compute means only for confident pixels
            denom = tf.reduce_sum(mask) + self.eps
            mean_conf = tf.reduce_sum(conf) / denom
            mean_var_class = tf.reduce_sum(var_c * mask) / denom
            
            mean_conf_list.append(mean_conf)
            mean_var_list.append(mean_var_class)
        
        return tf.stack(mean_conf_list), tf.stack(mean_var_list)
    
    def update_class_frequencies(self, pseudo_labels, valid_mask):
        """
        Update class frequencies using Exponential Moving Average (EMA)
        
        Args:
            pseudo_labels: [B, H, W] pseudo labels (0, 1, ..., C-1 or -1 for invalid)
            valid_mask: [B, H, W] boolean mask of valid predictions
        """
        pseudo_labels = tf.squeeze(pseudo_labels, axis=-1)
        valid_mask = tf.squeeze(valid_mask, axis=-1)
        
        # Set invalid labels to -1
        masked_labels = tf.where(valid_mask, pseudo_labels, 
                               tf.constant(-1, dtype=tf.float32))
        masked_labels = tf.cast(masked_labels, tf.int32)
        
        # One-hot encode valid pixels
        one_hot = tf.one_hot(tf.maximum(masked_labels, 0), depth=self.num_classes)
        valid_float_mask = tf.cast(masked_labels >= 0, tf.float32)
        one_hot = one_hot * tf.expand_dims(valid_float_mask, axis=-1)
        
        # Per-class confident pixel counts
        class_pixel_counts = tf.reduce_sum(one_hot, axis=[0, 1, 2])
        class_pixel_counts = tf.maximum(class_pixel_counts, self.eps)
        
        # Update frequencies with EMA
        total_confident_pixels = tf.reduce_sum(class_pixel_counts)
        current_batch_freq = class_pixel_counts / total_confident_pixels
        
        updated_freq = (self.lambda_decay * self.class_frequencies + 
                       (1 - self.lambda_decay) * current_batch_freq)
        self.class_frequencies.assign(updated_freq)
    
    def update_confidence_threshold(self, mean_pred_conf):
        """
        Update confidence threshold τ based on Equation (5) from paper:
        
        Args:
            mean_pred_conf: [C] mean confidence per class
        """
        # Compute imbalance term: η * (f_c)^m * log(1/(1 - f_c))
        log_term = tf.math.log(1.0 / (1.0 - self.class_frequencies + self.eps))
        imbalance_term = self.eta * tf.pow(self.class_frequencies, self.m) * log_term
        
        # Update threshold: τ_new = τ_old + α * ReLU(mean_conf - τ) + imbalance_term
        delta_tau = (self.alpha * tf.nn.relu(mean_pred_conf - self.tau) + 
                    imbalance_term)
        
        self.tau.assign_add(delta_tau)
        # Clamp between reasonable bounds
        self.tau.assign(tf.clip_by_value(self.tau, 0.7, 0.97))
    
    def update_uncertainty_threshold(self, mean_pred_var):
        """
        Update uncertainty threshold γ based on Equation (6) from paper:
        
        Args:
            mean_pred_var: [C] mean variance per class
        """
        # Compute imbalance term: ζ * (f_c)^n * log(1/(1 - f_c))
        log_term = tf.math.log(1.0 / (1.0 - self.class_frequencies + self.eps))
        imbalance_term = self.zeta * tf.pow(self.class_frequencies, self.n) * log_term
        
        # Update threshold: γ_new = γ_old - β * ReLU(γ - mean_var) - imbalance_term
        delta_gamma = (-self.beta * tf.nn.relu(self.gamma - mean_pred_var) - 
                      imbalance_term)
        
        self.gamma.assign_add(delta_gamma)
        # Clamp between reasonable bounds
        self.gamma.assign(tf.clip_by_value(self.gamma, 0.01, 0.2))
    
    def compute_class_weights(self):
        """
        Calculate class weights based on Equation (4) from paper:
        
        Returns:
            weights: [C] class weights
        """
        return 1.0 / tf.sqrt(self.class_frequencies + self.eps)
    
    def get_pseudo_mask(self, mean_conf, mean_var):
        """
        Generate pseudo-labels using adaptive thresholds (Equation 7):
        
        Args:
            mean_conf: [B, H, W, C] mean confidence
            mean_var: [B, H, W, C] mean variance
            
        Returns:
            pseudo_labels: [B, H, W, 1] with class indices or -1 for invalid
        """
        # Apply adaptive thresholds
        valid_mask = (mean_conf > self.tau) & (mean_var < self.gamma)
        
        # Choose the most confident class among valid ones
        masked_conf = tf.where(valid_mask, mean_conf, 
                             tf.constant(-1e9, dtype=mean_conf.dtype))
        hard_preds = tf.argmax(masked_conf, axis=-1)
        
        # Determine if any class was confident at each pixel
        has_valid_class = tf.reduce_any(valid_mask, axis=-1)
        
        # Assign -1 to invalid predictions
        pseudo_labels = tf.where(has_valid_class, hard_preds, 
                               -tf.ones_like(hard_preds, dtype=tf.int64))
        
        return tf.cast(tf.expand_dims(pseudo_labels, axis=-1), tf.float32)
    
    def adaptive_weighted_loss(self, y_true, y_pred, pseudo_mask, weights, loss_fn):
        """
        Compute adaptive weighted unsupervised loss (Equation 8):
        
        Args:
            y_true: Pseudo-labels [N, 1, 1, 1]
            y_pred: Predictions [N, 1, 1, 1]
            pseudo_mask: Confidence mask [N, 1, 1, 1]
            weights: Class weights [C]
            loss_fn: Base loss function
            
        Returns:
            Normalized adaptive weighted loss
        """
        # Normalize class weights
        weights = tf.cast(weights, tf.float32)
        weights = weights / (tf.reduce_sum(weights) + self.eps)
        
        # Compute per-pixel loss
        loss_per_pixel = loss_fn(y_true, y_pred)  # [N, 1, 1, 1]
        
        # Class labels
        y_class = tf.cast(tf.round(y_true), tf.int32)  # [N, 1, 1, 1]
        
        # Apply class weights to loss
        weighted_loss = 0.0
        for c in range(self.num_classes):
            # Create mask for class c
            class_mask = pseudo_mask * tf.cast(tf.equal(y_class, c), tf.float32)
            
            # Apply class weight
            weighted_loss += weights[c] * class_mask * loss_per_pixel
        
        # Normalize by number of confident pixels
        num_confident_pixels = tf.reduce_sum(pseudo_mask) + self.eps
        return tf.reduce_sum(weighted_loss) / num_confident_pixels
    
    def get_thresholds(self):
        """Get current threshold values"""
        return {
            'tau': self.tau.numpy(),
            'gamma': self.gamma.numpy(),
            'class_frequencies': self.class_frequencies.numpy()
        }
    
    def reset_thresholds(self):
        """Reset thresholds to initial values"""
        self.tau.assign([0.7] * self.num_classes)
        self.gamma.assign([0.2] * self.num_classes)
        self.class_frequencies.assign([1.0/self.num_classes] * self.num_classes)
