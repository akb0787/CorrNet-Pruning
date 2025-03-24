# main.py
import tensorflow as tf
from corrnet import preprocess_data, build_vgg16_model, correlation_based_pruning, evaluate_model, plot_history

# Set random seed
tf.random.set_seed(42)

if __name__ == "__main__":
    # Load and preprocess data
    x_train, y_train, x_test, y_test, datagen = preprocess_data()
    
    # Build and train baseline model
    baseline_model = build_vgg16_model()
    baseline_model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1, momentum=0.9),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
    baseline_model.fit(datagen.flow(x_train, y_train, batch_size=100),
                       epochs=10,
                       validation_data=(x_test, y_test))
    
    # Evaluate baseline
    print("Baseline Model Evaluation:")
    evaluate_model(baseline_model, x_test, y_test)
    
    # Perform pruning
    pruned_model, history = correlation_based_pruning(baseline_model, x_train, num_images=100, pruning_ratio=0.1)
    
    # Evaluate pruned model
    print("Pruned Model Evaluation:")
    evaluate_model(pruned_model, x_test, y_test)
    
    # Plot results
    plot_history(history)
