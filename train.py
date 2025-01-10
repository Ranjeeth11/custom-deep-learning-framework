def train(model, X_train, y_train, loss_fn, learning_rate, epochs):
    for epoch in range(epochs):
        # Forward pass
        predictions = model.forward(X_train)
        loss = loss_fn.forward(predictions, y_train)

        # Backward pass
        output_gradient = loss_fn.backward()
        model.backward(output_gradient, learning_rate)

        # Learning rate scheduling (decay the learning rate)
        if epoch % 20 == 0 and epoch > 0:
            learning_rate *= 0.9  # Reduce learning rate by 10% every 20 epochs

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss}, Learning Rate: {learning_rate}')
