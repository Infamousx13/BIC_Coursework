import numpy as np
import pandas as pd
import copy
#import sklearn
# Define Particle class
class Particle:
    def __init__(self, position, num_weights):
        self.position = position
        self.velocity = np.random.rand(num_weights)
        self.best_position = copy.copy(self.position)
        self.best_fitness = float('-inf')

# Neural Network class
class NeuralNetwork:
    def __init__(self, layer_sizes, activations):
        self.layers = len(layer_sizes) + 2
        self.layer_sizes = [4] + layer_sizes + [1]
        self.activations = activations
        self.weights = self.initialize_weights()

    def initialize_weights(self):
        weights = [np.random.uniform(-1, 1, (self.layer_sizes[i-1], self.layer_sizes[i])) for i in range(self.layers)]
        return weights

    def forward_pass(self, X):
        layer_output = X
        for i in range(self.layers-1):
            if layer_output.shape[1] != self.weights[i].shape[0]:
                layer_input = np.dot(layer_output, self.weights[i].T)
            else:     
                layer_input = np.dot(layer_output, self.weights[i])
            if i != 0:
                layer_output = self.apply_activation(layer_input, self.activations[i])
        layer_output = self.apply_activation(layer_output, self.activations[-1])
        return layer_output

    def apply_activation(self, x, activation):
        if activation == 'tanh':
            return np.tanh(x)
        elif activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif activation == 'relu':
            return np.maximum(0, x)
        else:
            raise ValueError("Invalid activation function")

    def update_weights(self, new_weights):
        start_index = 0
        for i in range(self.layers - 1):
            weight_shape = self.weights[i].shape
            reshaped_weights = np.reshape(new_weights[start_index:start_index + np.prod(weight_shape)], weight_shape)
            self.weights[i] = reshaped_weights  # Extract weights
            start_index += np.prod(weight_shape)

    def get_weights(self):
        return np.concatenate([self.weights[i].flatten() for i in range(self.layers)])
    
    def predict(self, X):
        y=[]
        y.append(self.forward_pass(X))
        return np.vstack(y)
    def loss(self, y_t, y_p):
        return np.mean((y_t - y_p) ** 2)

# PSO function
def particle_swarm_optimization(nn, num_particles, num_iterations, alpha, beta, gamma, delta, jump_size):
    num_weights = sum(np.prod(nn.weights[i].shape) for i in range(len(nn.weights)))
    particles = [Particle(nn.get_weights(), num_weights) for _ in range(num_particles)]
    
    global_best_position = np.concatenate([np.random.uniform(-1, 1, size=np.prod(nn.weights[i].shape))for i in range (nn.layers)])
    global_best_fitness = float('-inf')
    
    while (num_iterations !=0) :
        for particle in particles:
            current_fitness = np.sum(np.square(particle.position))
            if current_fitness > particle.best_fitness:
                particle.best_fitness = current_fitness
                particle.best_position = copy.copy(particle.position)
                #particle.best_position = particle.position.copy()
            if current_fitness > global_best_fitness:
                global_best_fitness = current_fitness
                global_best_position = copy.copy(particle.position)
                #global_best_position = particle.position.copy()
        for particle in particles:
            best_particle = particle.best_position
            if (len(particle.position)):
                best_info_index = np.random.choice(len(particle.position))
                best_informant = particle.position[best_info_index]
            else: 
                best_informant = particle.position
            best_any = global_best_position
            for particle in particles:
                b = np.random.uniform(0.0, beta)
                c = np.random.uniform(0.0, gamma)
                d = np.random.uniform(0.0, delta)
                particle.velocity = alpha * particle.velocity + b*(best_particle - particle.position) + c*(best_informant - particle.position) + d * (best_any - particle.position)
        for particle in particles:        
            particle.position += jump_size * particle.velocity
        num_iterations -=1
    print("Global best position: ",global_best_position)
    return global_best_position

# Load your dataset from CSV
data = pd.read_csv(r'C:\Users\infam\Desktop\Files\BIC HWU F21BC\banknote+authentication\data_banknote_authentication.csv')
rng = np.random.default_rng()
df = data.values
np.random.shuffle(df)

split = int(0.8 * data.shape[0])
X_train = df[:split, :4]
X_test = df[split:, :4]
y_train = df[:split, -1:]
y_test = df[split:, -1:] 

print("\n<<<<<<<<<<<<<<Neural Network>>>>>>>>>>>>>>\n")


layers = int(input("Enter the number of hidden layers: "))
layer_size = []
activation = []
for i in range(layers):
    layer_size.append(int(input(f"Enter the number of neurons in hidden layer {i+1}: ")))
    activation.append(input(f"Enter the activation function for the layer - sigmoid, tanh, relu, softmax {i+1}: "))
activation.append(input(f"Enter the activation function for the output layer - sigmoid, tanh, relu, softmax: "))
nn = NeuralNetwork(layer_sizes=layer_size, activations=activation)

# Run PSO to optimize neural network weights
#print("\n<<<<<<<<<<<<<<Particle Swarm>>>>>>>>>>>>>>>\n")
#num_particles = int(input("Enter the number of particles: "))
#num_iterations = int(input("Enter the number of iterations: "))
#alpha = float(input("Enter a value for alpha: "))
#beta = float(input("Enter a value for beta: "))
#gamma = float(input("Enter a value for gamma: "))
#delta = float(input("Enter a value for delta: "))
#jump_size = float(input("Enter a jump size: "))

num_particles = 50
num_iterations = 50
alpha = 0.5
beta = 1.5
gamma = 1.5
delta = 1.0
jump_size = 0.5

epochs = int(input("Enter the number of epochs: "))
for epoch in range(epochs):
    print (f"Epoch {epoch + 1}/{epochs}")
    nn.forward_pass(X_train)
    best_weights = particle_swarm_optimization(nn, num_particles, num_iterations, alpha, beta, gamma, delta, jump_size)
    nn.update_weights(best_weights)
    y_pred = nn.predict(X_test)
    y_hat = np.abs((y_pred > 0.5).astype(int))
    loss = nn.loss(y_test, y_hat)
    print("Loss =", loss)
    accuracy = np.mean(y_test==y_hat)
    print("Accuracy in this epoch = {:.2%}".format(accuracy))
