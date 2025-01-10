import numpy as np
import matplotlib.pyplot as plt
from layers import Dense, ReLU, Sigmoid, Dropout
from losses import MeanSquaredError
from model import Model
from train import train

def evaluate(model, X_test, y_test, loss_fn):
    predictions = model.forward(X_test)
    loss = loss_fn.forward(predictions, y_test)
    print(f"Test Loss: {loss}")

if __name__ == "__main__":
    # Sample Data
    X_train = np.random.rand(100, 2)  # 100 samples, 2 features
    y_train = np.random.rand(100, 1)  # 100 target values

    # Create model
    model = Model()
    model.add(Dense(2, 4))  # Input layer with 2 inputs and 4 neurons
    model.add(Sigmoid())     # Activation layer using Sigmoid
    model.add(Dropout(0.5))  # Add Dropout layer with 50% rate
    model.add(Dense(4, 1))   # Output layer with 1 output

    # Define loss function
    loss_fn = MeanSquaredError()
    learning_rate = 0.01

    # Train the model and collect losses for visualization
    epochs = 100
    losses = []  # List to store loss values for visualization
    for epoch in range(epochs):
        # Forward pass
        predictions = model.forward(X_train)
        loss = loss_fn.forward(predictions, y_train)
        losses.append(loss)  # Store loss value

        # Backward pass
        output_gradient = loss_fn.backward()
        model.backward(output_gradient, learning_rate)

        # Learning rate scheduling
        if epoch % 20 == 0 and epoch > 0:
            learning_rate *= 0.9

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss}, Learning Rate: {learning_rate}')

    # Plotting loss
    plt.plot(range(epochs), losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.show()

    # Evaluate the model
    X_test = np.random.rand(20, 2)  # Example test data
    y_test = np.random.rand(20, 1)   # Example test targets
    evaluate(model, X_test, y_test, loss_fn)
