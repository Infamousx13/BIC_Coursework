import numpy as np
import pandas as pd
import copy
#Particle Class
class Particle:
    #Initialize Particle
    def __init__(self, weights, num_weights):
        self.best_weights = weights
        self.position = np.random.rand(num_weights)
        self.velocity = np.random.rand(num_weights)
        self.best_position = copy.copy(self.position)
        self.particle_fitness = float('-inf')

# Neural Network class
class NeuralNetwork:
    #Initialize Neural Network
    def __init__(self, layer_sizes, activations):
        self.layers = len(layer_sizes) + 2
        self.layer_sizes = [4] + layer_sizes + [1]
        self.activations = activations
        self.weights = self.initialize_weights()
    #Initialize Weights
    def initialize_weights(self):
        weights = [np.random.uniform(-1, 1, (self.layer_sizes[i-1], self.layer_sizes[i])) for i in range(self.layers)]
        return weights
    #Forward Pass
    def forward_pass(self, X, y):
        layer_output = X
        for i in range(self.layers-1):
            if layer_output.shape[1] != self.weights[i].shape[0]:
                layer_input = np.dot(layer_output, self.weights[i].T)
            else:     
                layer_input = np.dot(layer_output, self.weights[i])
            if i != 0:
                layer_output = self.apply_activation(layer_input, self.activations[i])
        layer_output = self.apply_activation(layer_output, self.activations[-1])
        loss = self.loss(y, layer_output)
        return layer_output, loss
    #Activation Functions
    def apply_activation(self, x, activation):
        if activation == 'tanh':
            return np.tanh(x)
        elif activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif activation == 'relu':
            return np.maximum(0, x)
        else:
            raise ValueError("Invalid activation function")
    #Weight Update
    def update_weights(self, new_weights):
        start_index = 0
        for i in range(self.layers - 1):
            weight_shape = self.weights[i].shape
            reshaped_weights = np.reshape(new_weights[start_index:start_index + np.prod(weight_shape)], weight_shape)
            self.weights[i] = reshaped_weights
            start_index += np.prod(weight_shape)
    #Passing Weights
    def get_weights(self):
        return np.concatenate([self.weights[i].flatten() for i in range(self.layers)])
    #Prediction
    def predict(self, X):
        y=[]
        y.append(self.forward_pass(X))
        return np.vstack(y)
    #Loss Calculation using Mean Square Error Loss
    def loss(self, y_t, y_p):
        return np.mean((y_t - y_p) ** 2)

# Particle Swarm Optimizer: based on pseudocode in Essentials of Metaheuristics, P7
def particle_swarm_optimizer(nn, num_particles, num_iterations, alpha, beta, gamma, delta, jump_size, loss):
    #Weights sent as particles
    num_weights = sum(np.prod(nn.weights[i].shape) for i in range(len(nn.weights)))
    particles = [Particle(nn.get_weights(), num_weights) for _ in range(num_particles)]
    #Global Bests taken
    global_best_position = None
    global_best_fitness = None
    #Beginning loop
    while (num_iterations !=0) :
        for particle in particles:
            particle.particle_fitness = loss
            if (global_best_fitness == None) or (particle.particle_fitness > global_best_fitness):
                weight = particle.best_weights
                global_best_fitness = particle.particle_fitness
                global_best_position = particle.position
                #print("Temp GBP: ", global_best_position)
            #if current_fitness > particle.best_fitness:
                #print("Temp1 Fitness", current_fitness)
                #particle.best_fitness = current_fitness
                #particle.best_position = copy.copy(particle.position)
                #particle.best_position = particle.position.copy()
            #if current_fitness > global_best_fitness:
                #print("Temp2 Fitness: ", current_fitness)
                #global_best_fitness = current_fitness
                #global_best_position = copy.copy(particle.position)
                #global_best_position = particle.position.copy()
        for particle in particles:
            best_particle = particle.best_position
            if (len(particle.position)):
                best_info_index = np.random.choice(len(particle.position), 3)
                best_informant = np.concatenate([particle.position[x] for x in best_info_index])
            else: 
                best_informant = particle.position
            best_any = global_best_position
            dims = len(particles)
            for i in range(0,dims):
                b = np.random.uniform(0.0, beta)
                c = np.random.uniform(0.0, gamma)
                d = np.random.uniform(0.0, delta)
                particle.velocity[i] = alpha * particle.velocity[i] + b*(best_particle[i] - particle.position[i]) + c*(best_informant[i] - particle.position[i]) + d * (best_any[i] - particle.position[i])
        for particle in particles:        
            particle.position += jump_size * particle.velocity
        num_iterations -=1
    #print("Global best position: ",global_best_position)
    print("Weights: ", weight)
    return weight

def main():
    #Data loaded and preprocess
    data = pd.read_csv(r'C:\Users\infam\Desktop\Files\BIC HWU F21BC\banknote+authentication\data_banknote_authentication.csv')
    #rng = np.random.default_rng()
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
        activation.append(input(f"Enter the activation function for this layer - sigmoid, tanh, relu: "))
    activation.append(input(f"Enter the activation function for the output layer - sigmoid, tanh, relu: "))
    nn = NeuralNetwork(layer_sizes=layer_size, activations=activation)

    print("\n<<<<<<<<<<<<<<Particle Swarm>>>>>>>>>>>>>>>\n")
    #num_particles = int(input("Enter the number of particles: "))
    #num_iterations = int(input("Enter the number of iterations: "))
    #alpha = float(input("Enter a value for alpha: "))
    #beta = float(input("Enter a value for beta: "))
    #gamma = float(input("Enter a value for gamma: "))
    #delta = float(input("Enter a value for delta: "))
    #jump_size = float(input("Enter a jump size: "))

    num_particles = 50
    num_iterations = 50
    alpha = 0.9
    beta = 2.0
    gamma = 2.0
    delta = 0
    jump_size = 1

    epochs = int(input("Enter the number of epochs: "))
    for epoch in range(epochs):
        print (f"Epoch {epoch + 1}/{epochs}")
        y_pred_train, loss = nn.forward_pass(X_train, y_train)
        best_weights = particle_swarm_optimizer(nn, num_particles, num_iterations, alpha, beta, gamma, delta, jump_size, loss)
        nn.update_weights(best_weights)
        y_hat = np.abs((y_pred_train > 0.5).astype(int))
        print("Training Loss =", loss)
        accuracy = np.mean(y_train==y_hat)
        print("Training accuracy in this epoch = {:.2%}".format(accuracy))
        tp = np.sum(((y_train == 1)) & (y_pred_train == 1))
        fp = np.sum(((y_train == 0)) & (y_pred_train == 1))
        fn = np.sum(((y_train == 1)) & (y_pred_train == 0))
        precision = ((tp/tp+fp) if (tp + fp != 0) else 0)
        print("Precision: ", precision)
        recall = ((tp/tp+fn) if (tp+fn != 0 ) else 0)
        print("Recall: ", recall)
        f1score = ((2*(precision*recall)/(precision*recall) if precision+recall !=0 else 0))
        print("F1 Score: ", f1score)
    y_pred_test, lossx = nn.forward_pass(X_test, y_test)
    y_hatt = np.abs((y_pred_test > 0.5).astype(int))
    print("Testing Loss =", lossx)
    accuracyfinal = np.mean(y_test==y_hatt)
    print (np.vstack(y_hatt))
    print("Test accuracy = {:.2%}".format(accuracyfinal))
    tpfin = np.sum(((y_train == 1)) & (y_pred_train == 1))
    fpfin = np.sum(((y_train == 0)) & (y_pred_train == 1))
    fnfin = np.sum(((y_train == 1)) & (y_pred_train == 0))
    precision_fin = ((tpfin/tpfin+fpfin) if (tpfin + fpfin != 0) else 0)
    print("Precision: ", precision_fin)
    recall_fin = ((tpfin/tpfin+fnfin) if (tpfin+fnfin != 0 ) else 0)
    print("Recall: ", recall_fin)
    f1score_fin = ((2*(precision_fin*recall_fin)/(precision_fin*recall_fin) if (precision_fin+recall_fin) !=0 else 0))
    print("F1 Score: ", f1score_fin)
if __name__ == "__main__":
    main()
