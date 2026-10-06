import numpy as np


class MicroCluster:
    def __init__(self, lambd, creation_time):
        self.lambd = lambd
        self.decay_factor = 2 ** (-lambd)
        self.mean = 0
        self.variance = 0
        self.sum_of_weights = 0
        self.creation_time = creation_time
        self.we_ti = 0

    
    def update_weight(self,t):
        time_window = t-self.we_ti
        self.sum_of_weights = self.sum_of_weights * (self.decay_factor) ** time_window
        self.variance = self.variance * (self.decay_factor) ** time_window # <- Added line
        self.we_ti = t

    #og: def insert_sample(self, sample, weight):
    #new def insert_sample(self, sample, weight,t):
    def insert_sample(self, sample, weight,t):
        if self.sum_of_weights != 0:
            #new
            self.update_weight(t)
            # Update sum of weights
            old_sum_of_weights = self.sum_of_weights
            #new_sum_of_weights = old_sum_of_weights * self.decay_factor + weight
            #new
            new_sum_of_weights = old_sum_of_weights + weight

            # Update mean
            old_mean = self.mean
            new_mean = old_mean + \
                (weight / new_sum_of_weights) * (sample - old_mean)

            # Update variance
            old_variance = self.variance
            new_variance = old_variance * ((new_sum_of_weights - weight)
                                           / old_sum_of_weights) \
                + weight * (sample - new_mean) * (sample - old_mean)

            self.mean = new_mean
            self.variance = new_variance
            self.sum_of_weights = new_sum_of_weights
        else:
            self.mean = sample
            self.sum_of_weights = weight
            #new
            self.we_ti = t

    def radius(self):
        if self.sum_of_weights > 0:
            return np.linalg.norm(np.sqrt(self.variance / self.sum_of_weights))
        else:
            return float('nan')

    def center(self):
        return self.mean

    def weight(self):
        return self.sum_of_weights

    def __copy__(self):
        new_micro_cluster = MicroCluster(self.lambd, self.creation_time)
        new_micro_cluster.sum_of_weights = self.sum_of_weights
        new_micro_cluster.variance = self.variance
        new_micro_cluster.mean = self.mean
        new_micro_cluster.we_ti = self.we_ti 
        return new_micro_cluster