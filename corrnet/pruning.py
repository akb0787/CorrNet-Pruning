# corrnet/pruning.py
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from scipy.stats import pearsonr

def calculate_correlation(feature_maps):
    """Calculate Pearson correlation between successive feature maps."""
    num_filters = feature_maps.shape[-1]
    correlations = []
    
    for i in range(num_filters - 1):
        fm1 = feature_maps[:, :, :, i].flatten()
        fm2 = feature_maps[:, :, :, i + 1].flatten()
        corr, _ = pearsonr(fm1, fm2)
        correlations.append(corr)
    
    return np.array(correlations)

def select_filters_to_prune(feature_maps, threshold=0.9):
    """Select filters with high correlation for pruning."""
    correlations = calculate_correlation(feature_maps)
    filters_to_prune = np.where(correlations > threshold)[0]
    return filters_to_prune

def prune_layer(model, layer_idx, filters_to_prune):
    """Prune filters from a specific convolutional layer."""
    conv_layer = model.layers[layer_idx]
    weights = conv_layer.get_weights()
    w, b = weights[0], weights[1]
    
    keep_indices = [i for i in range(w.shape[-1]) if i not in filters_to_prune]
    new_w = w[:, :, :, keep_indices]
    new_b = b[keep_indices]
    
    new_layer = layers.Conv2D(
        filters=len(keep_indices),
        kernel_size=conv_layer.kernel_size,
        strides=conv_layer.strides,
        padding=conv_layer.padding,
        activation=conv_layer.activation,
        name=conv_layer.name + '_pruned'
    )
    
    new_model = models.Sequential()
    for i, layer in enumerate(model.layers):
        if i == layer_idx:
            new_model.add(new_layer)
        else:
            new_model.add(layer)
    
    new_model.layers[layer_idx].set_weights([new_w, new_b])
    return new_model

def correlation_based_pruning(model, x_train, num_images=100, pruning_ratio=0.1, epochs_per_iter=2):
    """Main pruning pipeline using correlation-based filter selection."""
    layer_outputs = [layer.output for layer in model.layers if isinstance(layer, layers.Conv2D)]
    intermediate_model = models.Model(inputs=model.input, outputs=layer_outputs)
    
    indices = np.random.choice(x_train.shape[0], num_images, replace=False)
    sample_images = x_train[indices]
    feature_maps = intermediate_model.predict(sample_images)
    
    for iter in range(3):
        print(f"Pruning iteration {iter + 1}")
        new_model = model
        
        for layer_idx, fm in enumerate(feature_maps):
            if fm.ndim == 4:
                filters_to_prune = select_filters_to_prune(fm, threshold=0.9)
                num_filters = fm.shape[-1]
                target_prune = int(num_filters * pruning_ratio)
                
                if len(filters_to_prune) > target_prune:
                    filters_to_prune = filters_to_prune[:target_prune]
                
                if len(filters_to_prune) > 0:
                    new_model = prune_layer(new_model, layer_idx, filters_to_prune)
                    print(f"Pruned {len(filters_to_prune)} filters from layer {layer_idx}")
        
        from .training import fine_tune_model
        history = fine_tune_model(new_model, x_train, y_train, x_test, y_test, datagen, epochs=epochs_per_iter)
        model = new_model
        feature_maps = intermediate_model.predict(sample_images)
    
    return model, history
