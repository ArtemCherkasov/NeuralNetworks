package ru.nn.mainclasses;

import java.util.ArrayList;

public class ForwardPropagation {
    private NeuralNetwork neuralNetwork;
    private ArrayList<NNLayer> nnLayersList;
    private int layersSize;

    public ForwardPropagation(NeuralNetwork neuralNetwork) {
        // TODO Auto-generated constructor stub
        this.neuralNetwork = neuralNetwork;
        this.nnLayersList = this.neuralNetwork.getNnLayersList();
        this.layersSize = this.neuralNetwork.getLayersSize();
    }

    public void step() {
        for (int layerIndex = 0; layerIndex < layersSize; layerIndex++) {
            this.nnLayersList.get(layerIndex).calculateNodeValueAndOutputVector();
        }
    }

}
