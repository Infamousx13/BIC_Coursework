import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# Define Particle class
class Particle:
    def __init__(self, num_weights):
        self.position = np.random.uniform(-1, 1, size=num_weights)
        self.velocity = np.random.rand(num_weights)
        self.best_position = self.position.copy()
        self.best_fitness = float('-inf')

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Define hyperbolic tangent activation function
def tanh_activation(x):
    return np.tanh(x)
def relu(x):
    return np.maximum(0,x)
    
# Neural Network class
class NeuralNetwork:
    def __init__(self):
        self.layers = int(input("Enter the number of layers: "))
        self.input_size = []
        self.activation = []

        for i in range(self.layers):
            input_size = int(input(f"Enter the input size for layer {i+1}: "))
            activation = input(f"Enter the activation function for layer {i+1} (tanh or sigmoid or relu): ")
            self.input_size.append(input_size)
            self.activation.append(activation)

        # Initialize weights with the last layer matching the number of features
        self.weights = [np.random.uniform(-1, 1, (self.input_size[i-1], self.input_size[i])) for i in range(1, self.layers-1)]
        # Last layer weights
        self.weights.append(np.random.uniform(-1, 1, (self.input_size[self.layers-2], 1)))

    def forward_pass(self, X, y_true):
        self.forward = self.predict(X)
        return np.mean(np.square(y_true - self.forward))

    def update_weights(self, new_weights):
        start_index = 0
        for i in range(self.layers - 1):
            weight_shape = self.weights[i].shape
            reshaped_weights = np.reshape(new_weights[start_index:start_index + np.prod(weight_shape)], weight_shape)
            self.weights[i] = reshaped_weights
            start_index += np.prod(weight_shape)

    def predict(self, X): 
        layer_output = X
        for i in range(self.layers - 1):
            layer_input = np.dot(layer_output, self.weights[i])
            if self.activation[i] == 'tanh':
                layer_output = tanh_activation(layer_input)
            elif self.activation[i] == 'sigmoid':
                layer_output = sigmoid(layer_input)
            elif self.activation[i] == 'relu':
                layer_output = relu(layer_input)
            else:
                raise ValueError("Invalid activation function")
        layer_output = sigmoid(layer_output)
        return layer_output

# PSO function
def particle_swarm_optimization(num_particles, num_iterations, alpha, beta, gamma, jump_size, X_train, y_train):
    num_weights = sum(np.prod(nn.weights[i].shape) for i in range(len(nn.weights)))
    particles = [Particle(num_weights) for _ in range(num_particles)]
    
    global_best_position = np.random.uniform(-1, 1, size=num_weights)
    global_best_fitness = float('-inf')
    
    for _ in range(num_iterations):
        for particle in particles:
            nn.update_weights(particle.position)
            current_fitness = nn.forward_pass(X_train, y_train)
            
            if current_fitness > particle.best_fitness:
                particle.best_fitness = current_fitness
                particle.best_position = particle.position.copy()
            
            if current_fitness > global_best_fitness:
                global_best_fitness = current_fitness
                global_best_position = particle.position.copy()
        
        for particle in particles:
            b = np.random.uniform(0.0, 1.0)
            c = np.random.uniform(0.0, 1.0)
            
            particle.velocity = alpha * particle.velocity + (b * beta)*(particle.best_position - particle.position) + (c * gamma)*(global_best_position - particle.position) 
            particle.position += jump_size * particle.velocity
        
    return global_best_position

# Load your dataset from CSV
data = pd.read_csv(r'C:\Users\infam\Desktop\Files\BIC HWU F21BC\banknote+authentication\data_banknote_authentication.csv')

df = data.values
np.random.shuffle(df)

split = int(0.8 * data.shape[0])
X_train = df[:split, :4]
X_test = df[split:, :4]
y_train = df[:split, -1:]
y_test = df[split:, -1:] 
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
# Create neural network
nn = NeuralNetwork()

# Run PSO to optimize neural network weights
num_particles = 100
num_iterations = 100
alpha = 0.9
beta = 2.0
gamma = 2.0
jump_size = 0.5

best_weights = particle_swarm_optimization(num_particles, num_iterations, alpha, beta, gamma, jump_size, X_train, y_train)

# Update neural network with the best weights found
nn.update_weights(best_weights)

# Usage of the predict method
predictions = nn.predict(X_test)
y_pred = (predictions > 0.5).astype(int)  # Convert probabilities to binary classes
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {accuracy:.2f}")