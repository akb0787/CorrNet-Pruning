# corrnet/evaluation.py
import tensorflow as tf

def evaluate_model(model, x_test, y_test):
    """Evaluate model performance."""
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
