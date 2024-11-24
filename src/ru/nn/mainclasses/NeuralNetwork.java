package ru.nn.mainclasses;

import ru.nn.helpers.DataHelper;

import java.util.ArrayList;

public class NeuralNetwork {

    private ArrayList<NNLayer> nnLayersList;
    private int layersSize;
    private double[] rightAnswer;
    private ForwardPropagation forwardPropagation;
    private BackPropagation backPropagation;
    private int inputAllUnitSize;
    private int singleUnitSize;

    public NeuralNetwork(int[] nodeCountList) {
        // TODO Auto-generated constructor stub
        this.layersSize = nodeCountList.length;
        this.nnLayersList = new ArrayList<NNLayer>();
        if (nodeCountList.length < 2) {
            System.out.println("error: the neural network cannot have less than two layers");
        } else {
            for (int layerIndex = 0; layerIndex < nodeCountList.length; layerIndex++) {
                NNLayer nnLayer;
                // zero layer inicialized
                if (layerIndex < 1) {
                    nnLayer = new NNLayer(0, nodeCountList[layerIndex], layerIndex);
                    nnLayer.setFirstLayer(true);
                } else {
                    nnLayer = new NNLayer(nodeCountList[layerIndex - 1], nodeCountList[layerIndex], layerIndex);
                    this.nnLayersList.get(layerIndex - 1).setNextLayer(nnLayer);
                    nnLayer.setPrevLayer(this.nnLayersList.get(layerIndex - 1));
                }
                this.nnLayersList.add(nnLayer);
            }
            this.nnLayersList.get(this.layersSize - 1).setLastLayer(true);
            this.rightAnswer = new double[nodeCountList[nodeCountList.length - 1]];
            this.forwardPropagation = new ForwardPropagation(this);
            this.backPropagation = new BackPropagation(this);
            //TEST DATA
            //this.nnLayersList.get(0).setNodesValueVector(GenerateDataHelper.getInputVectorTest(nodeCountList[0]));
            //GenerateDataHelper.setWeightMatrixTest(this);
            //this.rightAnswer = GenerateDataHelper.getRightAnswer();
        }
    }

    public void setRightAnswer(double[] rightAnswer) {
        this.rightAnswer = rightAnswer;
    }

    public double[] getRightAnswer() {
        return rightAnswer;
    }

    public ArrayList<NNLayer> getNnLayersList() {
        return nnLayersList;
    }

    public int getLayersSize() {
        return layersSize;
    }

    public ForwardPropagation getForwardPropagation() {
        return forwardPropagation;
    }

    public BackPropagation getBackPropagation() {
        return backPropagation;
    }

    public NNLayer getLastLayer() {
        return this.getNnLayersList().get(this.layersSize - 1);
    }

    public void setInputVector(double[] inputVector) {
        this.nnLayersList.get(0).setNodesValueVector(inputVector);
    }

    public double[] getNeuralNetworkAnswerNormalized() {
        double[] normalizedAnswer = new double[this.getLastLayer().getOutputVector().length];
        int pointerToIndex = this.inputAllUnitSize * this.singleUnitSize - 6;
        for (int indexOutputVector = 0; indexOutputVector < this.getLastLayer().getOutputVector().length; indexOutputVector++) {
            normalizedAnswer[indexOutputVector] = this.nnLayersList.get(0).getNodesValueVector()[pointerToIndex] - valueConverterSigmaYToX(this.getLastLayer().getOutputVector()[indexOutputVector]);
        }
        return normalizedAnswer;
    }

    public double valueConverterSigmaYToX(double y) {
        return Math.log(y / (1 - y));
    }

    public double getAverageQuadraticError() {
        double[] answer = this.getLastLayer().getOutputVector();
        double result = 0;
        for (int answerIndex = 0; answerIndex < answer.length; answerIndex++) {
            result = result + Math.pow(answer[answerIndex] - this.rightAnswer[answerIndex], 2);
        }
        return result / answer.length;
    }

    public String getAverageQuadraticErrorTextLine() {
        String outputState = "Average quadratic error()";
        double error = getAverageQuadraticError();
        outputState = String.join(" ", outputState, String.format("%2.10f", error));
        return outputState;
    }

    public void setTechnicalParameters(DataHelper dataHelper) {
        this.inputAllUnitSize = dataHelper.getInputUnitsSize();
        this.singleUnitSize = dataHelper.getSingleUnitSize();
    }

    public void saveWeightMatrixes(DataHelper dataHelper){
        dataHelper.saveWeightMatrixes(this);
    }

    public boolean loadWeightMatrixes(DataHelper dataHelper){
        return dataHelper.loadWeightMatrixes(this);
    }
}