import numpy as np
import matplotlib.pyplot as plt


class CostTrace(object):
    '''
    Object for keeping track of cost for each penalty term during optimization.

    '''

    def __init__(self, penalty_weights, SAMPLE_RATE=100):
        self.cost_sample_counter = 0
        self.cost_sample_rate = SAMPLE_RATE
        self.trace = {}

        for key in penalty_weights.keys():
            if penalty_weights[key] != 0:
                self.trace[key] = []

    def plot_trace(self):
        '''
        Plots the trace of the costs for each penalty term
        '''
        fig = plt.figure(figsize=(16, 16))
        ax = fig.add_subplot(111)
        for name, p_costs in self.trace.iteritems():
            ax.plot(p_costs, label=name)

        ax.legend()
        ax.set_title('Cost Trace Plot')
        ax.set_xlabel('Iteration * %d' % self.cost_sample_rate)
        ax.set_ylabel('Cost')

        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                         ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(20)

        fig.show()

    def add_penalty(self, key, p_cost):
        if self.cost_sample_counter % self.cost_sample_rate == 0:
            self.trace[key].append(p_cost)
            self.cost_sample_counter = 0

    def clear_trace(self):
        for key in self.trace.keys():
            self.trace[key] = []

    def inc_counter(self):
        self.cost_sample_counter += 1
