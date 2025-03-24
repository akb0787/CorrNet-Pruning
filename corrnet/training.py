# corrnet/training.py
import tensorflow as tf

def fine_tune_model(model, x_train, y_train, x_test, y_test, datagen, epochs=10):
    """Fine-tune the pruned model."""
    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit(datagen.flow(x_train, y_train, batch_size=100),
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        verbose=1)
    return history
