import numpy as np
import pandas as pd
def relu(x):
    return np.maximum(x, 0)
def sigmoid(x):
    return 1 / (1 + np.exp(x))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def tanh(x):
    return np.tanh(x)

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

class Model:

    def __init__(self):
        self.layers = []
        self.n_wts = []
        self.compiled = False

    def add_layer(self, layer):
        self.layers.append(layer)
        self.n_wts.append(layer.n_wts)

    def _calc_dims(self):
        return int(np.sum(self.n_wts))

    def compile(self, loss_fn, dataloader_cls, metric_fn=None, c1=2.,
                c2=2., n_workers=10, batch_size=32, batches_per_epoch=100,
                position_range=(-1, 1), velocity_range=(-1, 1)):

        self.dataloader_cls = dataloader_cls
        self.data_loader = partial(dataloader_cls, batch_size=batch_size,
                                  repeat=True, shuffle=True)
        self.metric_fn = metric_fn

        self.loss_fn = partial(loss_fn, layers=self.layers, n_wts=self.n_wts, dims=self._calc_dims())

        self.optimizer = partial(ParticleSwarmOptimizer, particle_cls=Particle, c1=c1,
            c2=c2, n_particles=n_workers, compare_fn=lambda x, y: x < y,
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
        if isinstance(self.optimizer, partial):
            self.optimizer = self.optimizer(fitness_fn=loss_fn)

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
class Particle:

    def __init__(self, random, position=[0.],
                 velocity=[0.], position_range=None,
                 velocity_range=None, dims=None, alpha=0.1):
        self._validate(random, position, velocity, position_range, velocity_range, dims, alpha)

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
            self.position = np.asarray(position)
            self.velocity = np.asarray(velocity)
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
class ParticleSwarmOptimizer:

    def __init__(self, particle_cls, c1, c2, n_particles,
                 fitness_fn, compare_fn, n_iter=1, dims=None,
                 random=True, particles_list=None, position_range=None,
                 velocity_range=None):
        self._validate(particle_cls, c1, c2, n_particles,
                       fitness_fn, compare_fn, n_iter, dims,
                       random, particles_list)

        self.particle_cls = particle_cls
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
                particle = self.particle_cls(self.random, position_range=self.position_range,
                                             velocity_range=self.velocity_range, dims=self.dims)
                self.particles_list.append(particle)

        self.gbest = self.particles_list[0].position
        self._update_gbest()

        self.dims = self.particles_list[0].dims
