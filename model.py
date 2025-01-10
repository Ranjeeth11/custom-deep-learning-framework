from layers import Dense, ReLU  # Correctly import Dense and ReLU classes

class Model:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backward(self, output_gradient, learning_rate):
        for layer in reversed(self.layers):
            if isinstance(layer, Dense):
                output_gradient = layer.backward(output_gradient, learning_rate)
            else:
                output_gradient = layer.backward(output_gradient)
