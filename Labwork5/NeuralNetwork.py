import random
import math
from typing import List

class Neuron:
    def __init__(self, activation) -> None:
        self.output = 0
        self.expectedOutput = 0
        self.activation = activation

    def update(self, input):
        self.output = self.activation(input)

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
        self.newWeight = weight

class Layer:
    def __init__(self, activation, neuron_no) -> None:
        self.activation = activation
        self.neurons = self._init_neurons(neuron_no)

    def _init_neurons(self, neuron_no):
        neurons = [BiasNeuron()]
        for _ in range(neuron_no):
            neurons.append(Neuron(self.activation))
        return neurons

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
                if (isinstance(tNeuron, BiasNeuron)): continue
                links.append(Link(fNeuron, tNeuron, 0))#random.randrange(-100,100)/100))
        return links

class NeuronNetwork:
    def __init__(self, activation, learningRate, file) -> None:
        layer_no, neuron_no = self._get_data(file)
        self.activation = activation
        self.learningRate = learningRate
        self.layers = self._init_layers(layer_no, neuron_no)
        self.layerLinks = self._init_layer_links()

    def _get_data(self, file: str):
        return 3, [2,2,1]

    def _init_layers(self, layer_no, neuron_no):
        layers = []
        for i in range(layer_no):
            layers.append(Layer(self.activation, neuron_no[i]))
        return layers

    def _init_layer_links(self):
        layerLinks = []
        for i in range(len(self.layers)-1):
            layerLinks.append(LayerLink(self.layers[i], self.layers[i+1]))
        return layerLinks

    def train(self, X, Y):
        for layer in self.layers[::-1]:
            for layerLink in self.layerLinks:
                if layerLink.toLayer == layer:
                    currentLayerLink = layerLink
        
            for neuron in layer.neurons:
                if isinstance(neuron, BiasNeuron): continue
                # neuron.expectedOutput = self.activation(self.weighted_sum(neuron, currentLayerLink))
                for link in currentLayerLink.links:
                    # for x, y in zip(X, Y):
                    #     self.predict(x)
                    link.newWeight = self.update_weight(link, neuron, currentLayerLink, X, Y)

        self.update_model()

    def update_weight(self, link, neuron, currentLayerLink, X, Y):
        error = 0
        for x, y in zip(X, Y):
            self.predict(x)
            neuron.expectedOutput = self.activation(self.weighted_sum(neuron, currentLayerLink))
            error += -self.gradient_descent(neuron.expectedOutput, neuron.output, link.fromNeuron.output)

        return link.weight - self.learningRate * error/len(X)

    def gradient_descent(self, expectedY, y, x):
        return -(y*(1-expectedY) + expectedY*(1-y)) * x

    def weighted_sum(self, neuron, layerLink):
        weightedSum = 0
        for link in layerLink.links:
            if neuron == link.toNeuron:
                weightedSum += link.weight * link.fromNeuron.output
        return weightedSum
    
    def update_model(self):
        for i, layer in enumerate(self.layers):
            for j, neuron in enumerate(layer.neurons):
                if isinstance(neuron, BiasNeuron): continue
                neuron.output = neuron.expectedOutput

        for i, layerLink in enumerate(self.layerLinks):
            for j, link in enumerate(layerLink.links):
                link.weight = link.newWeight
                print(f"LayerLink {i} Link {j} weight = {neuron.output}")
                
    def predict(self, inputs):
        input_layer = self.layers[0]
        for i, neuron in enumerate(input_layer.neurons[1:]):
            neuron.output = inputs[i]

        for i, layer in enumerate(self.layers[1:]):
            for neuron in layer.neurons:
                # TODO Change to use weighted_sum function
                neuron_input = 0
                for link in self.layerLinks[i].links:
                    if link.toNeuron == neuron:
                        neuron_input += link.weight * link.fromNeuron.output
                neuron.update(neuron_input)
        
        predict = [neuron.output for neuron in self.layers[-1].neurons[1:]]
        return predict

    def output(self):
        for i in range(len(self.layers)):
            for neuron in self.layers[i].neurons:
                print(f"{i} layer: {neuron.output}")

if __name__ == "__main__":
    activation = lambda x: 1/(1+math.exp(-x))
    learningRate = 0.1
    X = [[1,1], [1,0], [0,1], [0,0]]
    Y = [0, 1, 1, 0]
    nn = NeuronNetwork(activation, learningRate, "")
    
    for i in range(2):
        print(f"Epoch {i}")
        nn.train(X, Y)
        print(f"Predict {nn.predict([0,0])}")
        print("\n")


    
