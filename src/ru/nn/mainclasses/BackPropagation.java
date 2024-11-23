package ru.nn.mainclasses;

import java.util.ArrayList;

public class BackPropagation {
    private NeuralNetwork neuralNetwork;
    private ArrayList<NNLayer> nnLayersList;
    private int layersSize;

    public BackPropagation(NeuralNetwork neuralNetwork) {
        // TODO Auto-generated constructor stub
        this.neuralNetwork = neuralNetwork;
        this.nnLayersList = this.neuralNetwork.getNnLayersList();
        this.layersSize = this.neuralNetwork.getLayersSize();
    }

    public void calculateLastLayerErrors() {
        NNLayer lastLayer = this.neuralNetwork.getLastLayer();
        lastLayer.calculateErrosVector(this.neuralNetwork.getRightAnswer());
    }

    public void calculateLastLayer_dE_dOut_NetworkOut() {
        NNLayer lastLayer = this.neuralNetwork.getLastLayer();
        lastLayer.calculate_dE_dOut_NetworkOut(this.neuralNetwork.getRightAnswer());
    }

    public void calculateLayer_dE_dOut(int layerIndex) {
        NNLayer lastLayer = this.neuralNetwork.getNnLayersList().get(layerIndex);
        lastLayer.calculate_dE_dOut();
    }

    public void calculateLastLayer_dOut_dNet() {
        NNLayer lastLayer = this.neuralNetwork.getLastLayer();
        lastLayer.calculate_dOut_dNet();
    }

    public void calculateLayer_dOut_dNet(int layerIndex) {
        NNLayer layer = this.neuralNetwork.getNnLayersList().get(layerIndex);
        layer.calculate_dOut_dNet();
    }

    public void calculateLastLayer_dNet_dw() {
        NNLayer lastLayer = this.neuralNetwork.getLastLayer();
        lastLayer.calculate_dNet_dw();
    }

    public void calculateLayer_dNet_dw(int layerIndex) {
        NNLayer lastLayer = this.neuralNetwork.getNnLayersList().get(layerIndex);
        lastLayer.calculate_dNet_dw();
    }

    public void calculateLastLayer_dEtotal_dw() {
        NNLayer lastLayer = this.neuralNetwork.getLastLayer();
        lastLayer.calculate_dEtotal_dw();
    }

    public void calculateLayer_dEtotal_dw(int layerIndex) {
        NNLayer lastLayer = this.neuralNetwork.getNnLayersList().get(layerIndex);
        lastLayer.calculate_dEtotal_dw();
    }

    public void correctLastLayerWeightMatrix() {
        NNLayer lastLayer = this.neuralNetwork.getLastLayer();
        lastLayer.correctWeightMatrix();
    }

    public void correctLayerWeightMatrix(int layerIndex) {
        NNLayer lastLayer = this.neuralNetwork.getNnLayersList().get(layerIndex);
        lastLayer.correctWeightMatrix();
    }

    public void step() {
        this.calculateLastLayerErrors();
        this.calculateLastLayer_dE_dOut_NetworkOut();
        this.calculateLastLayer_dOut_dNet();
        this.calculateLastLayer_dNet_dw();
        this.calculateLastLayer_dEtotal_dw();
        for (int layerIndex = this.neuralNetwork.getLayersSize() - 2; layerIndex > 0; layerIndex--) {
            this.calculateLayer_dE_dOut(layerIndex);
            this.calculateLayer_dOut_dNet(layerIndex);
            this.calculateLayer_dNet_dw(layerIndex);
            this.calculateLayer_dEtotal_dw(layerIndex);
        }
        this.correctLastLayerWeightMatrix();
        for (int layerIndex = this.neuralNetwork.getLayersSize() - 2; layerIndex > 0; layerIndex--) {
            this.correctLayerWeightMatrix(layerIndex);
        }
    }

}