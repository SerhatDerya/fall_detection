import data_handler
import my_models
import matplotlib.pyplot as plt

# Load training and validation data
X_train, y_train = data_handler.Fall_dataset('train').getAll()
X_test, y_test = data_handler.Fall_dataset('val').getAll()

# Initialize the fall detection model
model = my_models.fall_detection_model()

# Train the model and capture the training history
history = model.train(X_train, y_train)

# Plot the training and validation loss
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model
model.evaluate(X_test, y_test)

# Save the trained model weights
model.save("fall_detection_model_14.weights.h5")
