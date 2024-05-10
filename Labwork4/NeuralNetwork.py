import random
import math
from typing import List

class Neuron:
    def __init__(self, activation) -> None:
        self.output = 0
        self.activation = activation

    def update(self, input):
        self.output = self.activation(input)

    def forward(self):
        return self.output

class BiasNeuron(Neuron):
    def __init__(self) -> None:
        self.output = 1

    def update(self, input):
        pass

class Link:
    def __init__(self, fromNeuron: Neuron, toNeuron: Neuron, weight: int) -> None:
        self.fromNeuron = fromNeuron
        self.toNeuron = toNeuron
        self.weight = weight

class Layer:
    def __init__(self, neuron_no) -> None:
        self.neurons = self._init_neurons(neuron_no)

    def _init_neurons(self, neuron_no):
        neurons = [BiasNeuron()]
        for _ in range(neuron_no):
            neurons.append(Neuron())

    def output(self):
        pass

class LayerLink:
    def __init__(self, fromLayer: Layer, toLayer: Layer) -> None:
        self.fromLayer = fromLayer
        self.toLayer = toLayer
        self.links = self._init_links()

    def _init_links(self):
        links = []
        for fNeuron in self.fromLayer.neurons:
            for tNeuron in self.toLayer.neurons:
                if (isinstance(tNeuron, BiasNeuron)): 
                    pass
                links.append(Link(fNeuron, tNeuron, random.random()))
        return links

class NeuronNetwork:
    def __init__(self, file) -> None:
        layer_no, neuron_no = self._get_data(file)
        self.layers = self._init_layers(layer_no, neuron_no)
        self.layerLinks = self._init_layer_links()

    def _get_data(self, file: str):
        return 3, [2,2,1]

    def _init_layers(self, layer_no, neuron_no):
        layers = []
        for i in range(layer_no):
            layers.append(Layer(neuron_no[i]))
        return layers

    def _init_layer_links(self):
        layerLinks = []
        for i in range(len(self.layers)-1):
            layerLinks.append(self.layers[i], self.layers[i+1])
        return layerLinks

    def input(self, inputs: list):
        for i in inputs:
            self.layers[0].neurons[i].output = inputs[i]

    def output(self):
        for i in range(len(self.layers[1:])):
            neuron_input = 0
            for neuron in self.layers[i]:
                for link in self.layerLinks[i]:
                    if link.toNeuron == neuron:
                        pass


if __name__ == "__main__":
    activation = lambda x: 1/(math.exp(-x))
