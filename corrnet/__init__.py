# corrnet/__init__.py
"""CorrNet: Pearson Correlation Based Pruning for Efficient CNNs."""
__version__ = "1.0.0"

from .data_preprocessing import preprocess_data
from .model_builder import build_vgg16_model
from .pruning import calculate_correlation, select_filters_to_prune, prune_layer, correlation_based_pruning
from .training import fine_tune_model
from .evaluation import evaluate_model
from .utils import plot_history
