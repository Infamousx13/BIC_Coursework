import numpy as np
import pandas as pd
from copy import deepcopy
class DataLoader:

    def __init__(self, X, y=None, batch_size=32, repeat=True, shuffle=True):
        self.X = deepcopy(X)
        self.y = deepcopy(y)

        if self.y is not None and len(self.y.shape) == 1:
            self.y = np.expand_dims(self.y, axis=1)

        self.batch_size = batch_size
        self.repeat = repeat
        self.shuffle = shuffle

    def _shuffle(self):
        if self.y is not None:
            indices = np.arange(0, self.X.shape[0])
            np.random.shuffle(indices)
            self.X = self.X[indices]
            self.y = self.y[indices]
        else:
            np.random.shuffle(self.X)

    def preprocess(self, X, y=None):
        if y is None:
            return X
        return X, y

    def _inf_generator(self):
        m = self.X.shape[0]
        while True:
            if self.shuffle:
                self._shuffle()

            for i in range(0, m, self.batch_size):
                if i + self.batch_size > m:
                    t_index = i
                    b_index = self.batch_size-(m-i)
                    X = np.vstack((self.X[t_index:], self.X[:b_index]))
                    
                    if self.y is not None:
                        y = np.vstack((self.y[t_index:], self.y[:b_index]))
                        yield self.preprocess(X, y)
                    else:
                        yield self.preprocess(X)
                else:
                    X = self.X[i:i+self.batch_size]

                    if self.y is not None:
                        yield self.preprocess(X, self.y[i:i+self.batch_size])
                    else:
                        yield self.preprocess(X)
    def get_generator(self):
        return self._inf_generator()
def relu(x):
    return np.maximum(x, 0)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def tanh(x):
    return np.tanh(x)

def accuracy(y, y_hat):
    y = np.argmax(y, axis=1)
    y_hat = np.argmax(y_hat, axis=1)

    return np.mean(y==y_hat)

class Layer:

    def __init__(self, in_units, units, activation):
        self.w_shape = (in_units, units)
        self.b_shape = (1, units)
        self.n_wts = in_units * units + units
        self.shape = (-1, units)
        self.activation = activation

    def _reshape_weights(self, wts):
        W = np.reshape(wts[:self.w_shape[0] * self.w_shape[1]], self.w_shape)
        b = np.reshape(wts[self.w_shape[0] * self.w_shape[1]:], self.b_shape)
        return W, b

    def forward(self, wts, x):
        W, b = self._reshape_weights(wts)
        return self.activation(np.dot(x, W) + b)
class Loss:

    def __init__(self, data_loader, layers, n_wts, dims):
        self.data_loader = data_loader
        self.layers = layers
        self.n_wts = n_wts
        self.dims = dims

    def _forward(self, wts):
        w_index = 0
        X, y = next(self.data_loader)
        for i, layer in enumerate(self.layers):
            X = layer.forward(wts[w_index:w_index+self.n_wts[i]], X)
            w_index += self.n_wts[i]
        return y, X
    
class MSELoss(Loss):

    def _loss(self, y, y_hat):
        return np.mean((y - y_hat) ** 2)

    def __call__(self, wts):
        y, y_hat = self._forward(wts)
        return self._loss(y, y_hat)

class RMSELoss(Loss):

    def _loss(self, y, y_hat):
        return np.sqrt(np.mean((y - y_hat) ** 2))

    def __call__(self, wts):
        y, y_hat = self._forward(wts)
        return self._loss(y, y_hat)

class Particle:

    def __init__(self, random, position=[0.],
                 velocity=[0.], position_range=None,
                 velocity_range=None, dims=None, alpha=0.1):

        self.random = random
        self.position = position
        self.velocity = velocity
        self.position_range = position_range
        self.velocity_range = velocity_range
        self.dims = dims
        self.alpha=alpha

        self._init_particle()

        self.pbest = self.position
    def _init_particle(self):
        if self.random:
            self.position = np.random.uniform(low=self.position_range[0],
                                              high=self.position_range[1],
                                              size=(self.dims,))
            self.velocity = np.random.uniform(low=-abs(self.velocity_range[1]-self.velocity_range[0]),
                                              high=abs(self.velocity_range[1]-self.velocity_range[0]),
                                              size=(self.dims,))
        else:
            self.position = np.asarray(self.position)
            self.velocity = np.asarray(self.velocity)
            self.dims = self.position.shape[0]
    def update(self, c1, c2, gbest, fitness_fn, compare_fn):
        self._update_velocity(c1, c2, gbest)
        self._update_position(fitness_fn, compare_fn)
    def _update_velocity(self, c1, c2, gbest):
        self.alpha = self.alpha / 2
        wrt_pbest = c1 * np.random.rand() * (self.pbest - self.position)
        wrt_gbest = c2 * np.random.rand() * (gbest - self.position)
        self.velocity = self.alpha * self.velocity + wrt_pbest + wrt_gbest

    def _update_position(self, fitness_fn, compare_fn):
        self.position = self.position + self.velocity + 0.01 * self.position
        if compare_fn(fitness_fn(self.position), fitness_fn(self.pbest)):
            self.pbest = self.position

    def __repr__(self):
        return '<Particle: dims={} random={}>'.format(self.dims, self.random)

class ParticleSwarmOptimizer(Particle):

    def __init__(self, c1, c2, n_particles,
                 fitness_fn=None, compare_fn=None, n_iter=1, dims=None,
                 random=True, particles_list=None, position_range=None,
                 velocity_range=None):

        self.c1 = c1
        self.c2 = c2
        self.n_particles = n_particles
        self.n_iter = n_iter
        self.fitness_fn = fitness_fn
        self.compare_fn = compare_fn
        self.position_range = position_range
        self.velocity_range = velocity_range
        self.dims = dims
        self.random = random
        self.particles_list = particles_list

        self._init_particles_list()
    def _get_fitness(self, position):
        return self.fitness_fn(position)

    def _update_gbest(self):
        for particle_i in self.particles_list:
            l1, l2 = self._get_fitness(particle_i.pbest), self._get_fitness(self.gbest)
            print(l1)
            if self.compare_fn(l1, l2):
                self.gbest = particle_i.position

    def _init_particles_list(self):
        if self.random:
            self.particles_list = []

            for i in range(self.n_particles):
                particle = Particle(self.random, position_range=self.position_range, velocity_range=self.velocity_range, dims=self.dims)
                self.particles_list.append(particle)

        self.gbest = self.particles_list[0].position
        self._update_gbest()

        self.dims = self.particles_list[0].dims
    def optimize(self):
        for particle in self.particles_list:
            particle.update(self.c1, self.c2, self.gbest, self.fitness_fn, self.compare_fn)
        self._update_gbest()
        return self
class Model(DataLoader, Loss, ParticleSwarmOptimizer):

    def __init__(self):
        self.layers = []
        self.n_wts = []
        self.compiled = False

    def add_layer(self, layer):
        self.layers.append(layer)
        self.n_wts.append(layer.n_wts)

    def _calc_dims(self):
        return int(np.sum(self.n_wts))

    def compile(self, X, y, loss_fn, metric_fn=None, c1=2.,
                c2=2., n_workers=10, batch_size=32, batches_per_epoch=100,
                position_range=(-1, 1), velocity_range=(-1, 1)):

        self.data_loader = DataLoader(X,y,batch_size=batch_size,
                                  repeat=True, shuffle=True)
        self.metric_fn = metric_fn

        self.loss_fn = Loss(data_loader=self.data_loader,layers=self.layers, n_wts=self.n_wts, dims=self._calc_dims())

        self.optimizer =ParticleSwarmOptimizer(c1=c1,
            c2=c2, n_particles=n_workers, fitness_fn=None,compare_fn=lambda x, y: x < y,
            n_iter=batches_per_epoch, dims=self._calc_dims(), random=True,
            position_range=position_range, velocity_range=velocity_range)

        self.compiled = True

    def _forward(self, X, wts):
        w_index = 0
        for i, layer in enumerate(self.layers):
            X = layer.forward(wts[w_index:w_index+self.n_wts[i]], X)
            w_index += self.n_wts[i]
        return X

    def fit(self, X, y, epochs=1):
        assert self.compiled, 'Call compile before training'

        data_loader = self.data_loader(X=X, y=y).get_generator()
        loss_fn = self.loss_fn(data_loader=data_loader)
        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch+1, epochs))
            self.optimizer.optimize()
            if self.metric_fn is not None:
                print('Metric: {}'.format(self.metric_fn(y, self.predict(X))), end='\t')
            print('Loss:', self._loss(X, y))

    def predict(self, X):
        assert self.compiled, 'Call compile before Prediction'

        data_loader = self.dataloader_cls(X,
            batch_size=32, repeat=False, shuffle=False).get_generator()
        y = []
        for X in data_loader:
            y.append(self._forward(X, self.optimizer.gbest))
        return np.vstack(y)

    def _loss(self, X, y):
        data_loader = self.dataloader_cls(X, y,
            batch_size=32, repeat=False, shuffle=False).get_generator()
        loss_fn = self.loss_fn(data_loader=data_loader)
        y = []
        try:
            while True:
                y.append(loss_fn(self.optimizer.gbest))
        except StopIteration:
            return np.mean(y)

df = pd.read_csv(r'C:\Users\infam\Desktop\Files\BIC HWU F21BC\banknote+authentication\data_banknote_authentication.csv')
data = df.values
np.random.shuffle(data)

split = int(0.8 * data.shape[0])
x_train = data[:split, :3]
x_test = data[split:, :3]
y_train = data[:split, 4]
y_test = data[split:, 4]

model = Model()
model.add_layer(Layer(1372, 4, relu))
model.add_layer(Layer(4, 10, sigmoid))
model.add_layer(Layer(10, 10, tanh))
model.add_layer(Layer(10, 1, softmax))

model.compile(x_train,y_train,MSELoss, accuracy, c1=1, c2=1, n_workers=50, batch_size=32, batches_per_epoch=100, position_range=(-1,1), velocity_range=(-1,1))
model.fit(x_train, y_train, 10)
y_hat = model.predict(x_test)

print('Accuracy:', accuracy(y_test,y_hat))