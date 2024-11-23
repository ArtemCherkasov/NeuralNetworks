package ru.nn.helpers;

import ru.nn.mainclasses.NNLayer;
import ru.nn.mainclasses.NeuralNetwork;

import java.util.Date;

public class OutputTextHelper {
    public static String getStateText(NeuralNetwork neuralNetwork) {
        String outputHeadFoot = "OutputTextHelper.java #######################################################";
        String outputState = outputHeadFoot.concat(System.lineSeparator());
        outputState = outputState.concat("Neural networks state").concat(System.lineSeparator());
        for (int layerIndex = 0; layerIndex < neuralNetwork.getLayersSize(); layerIndex++) {
            NNLayer nnLayer = neuralNetwork.getNnLayersList().get(layerIndex);
            outputState = outputState.concat(nnLayer.getLayerNameAndSize()).concat(System.lineSeparator());
            outputState = outputState.concat(nnLayer.getNodesVectorTextline()).concat(System.lineSeparator());
            outputState = outputState.concat(nnLayer.getWeightMatrixText());
            outputState = outputState.concat(nnLayer.getOutputVectorTextline()).concat(System.lineSeparator());
        }
        outputState = outputState.concat(new Date().toString()).concat(System.lineSeparator()).concat(outputHeadFoot);
        outputState = outputState.concat(System.lineSeparator());
        return outputState;
    }

    public static String getLastLayerErrorsText(NeuralNetwork neuralNetwork) {
        return neuralNetwork.getLastLayer().getErrorOutputVectorText();
    }

    public static String getLastLayerErrorTotalText(NeuralNetwork neuralNetwork) {
        return neuralNetwork.getLastLayer().getErrorTotalText();
    }

    public static String get_dE_dOut_Text(NeuralNetwork neuralNetwork) {
        return neuralNetwork.getLastLayer().get_dE_dOut_Text();
    }

    public static String get_dE_dOut_Text(NeuralNetwork neuralNetwork, int layer_index) {
        return neuralNetwork.getNnLayersList().get(layer_index).get_dE_dOut_Text();
    }

    public static String get_dOut_dNet_Text(NeuralNetwork neuralNetwork) {
        return neuralNetwork.getLastLayer().get_dOut_dNet_Text();
    }

    public static String get_dOut_dNet_Text(NeuralNetwork neuralNetwork, int layer_index) {
        return neuralNetwork.getNnLayersList().get(layer_index).get_dOut_dNet_Text();
    }

    public static String get_dNet_dw_Text(NeuralNetwork neuralNetwork) {
        return neuralNetwork.getLastLayer().get_dNet_dw_Text();
    }

    public static String get_dEtotal_dw_Text(NeuralNetwork neuralNetwork) {
        return neuralNetwork.getLastLayer().get_dEtotal_dw_Text();
    }

    public static String get_dEtotal_dw_Text(NeuralNetwork neuralNetwork, int layer_index) {
        return neuralNetwork.getNnLayersList().get(layer_index).get_dEtotal_dw_Text();
    }

    public static String getLastLayerWeightMatrixText(NeuralNetwork neuralNetwork) {
        return neuralNetwork.getLastLayer().getWeightMatrixText();
    }

    public static String getWeightMatrixText(NeuralNetwork neuralNetwork, int layer_index) {
        return neuralNetwork.getNnLayersList().get(layer_index).getWeightMatrixText();
    }

    public static String getBackPropagationInfo(NeuralNetwork neuralNetwork) {
        String outputHeadFoot = "Back propagation info() #######################################################";
        String outputState = outputHeadFoot.concat(System.lineSeparator());
        outputState = outputState.concat("getLastLayerErrorsText()");
        outputState = outputState.concat(System.lineSeparator());
        outputState = outputState.concat(getLastLayerErrorsText(neuralNetwork));
        outputState = outputState.concat(System.lineSeparator());
        outputState = outputState.concat(getLastLayerErrorTotalText(neuralNetwork));
        outputState = outputState.concat(System.lineSeparator());
        outputState = outputState.concat(get_dE_dOut_Text(neuralNetwork));
        outputState = outputState.concat(System.lineSeparator());
        outputState = outputState.concat(get_dOut_dNet_Text(neuralNetwork));
        outputState = outputState.concat(System.lineSeparator());
        outputState = outputState.concat(get_dNet_dw_Text(neuralNetwork));
        outputState = outputState.concat(System.lineSeparator());
        outputState = outputState.concat(get_dEtotal_dw_Text(neuralNetwork));
        outputState = outputState.concat(System.lineSeparator());
        outputState = outputState.concat(getLastLayerWeightMatrixText(neuralNetwork));
        outputState = outputState.concat(System.lineSeparator());
        outputState = outputState.concat(get_dE_dOut_Text(neuralNetwork, 1));
        outputState = outputState.concat(System.lineSeparator());
        outputState = outputState.concat(get_dOut_dNet_Text(neuralNetwork, 1));
        outputState = outputState.concat(System.lineSeparator());
        outputState = outputState.concat(get_dEtotal_dw_Text(neuralNetwork, 1));
        outputState = outputState.concat(System.lineSeparator());
        outputState = outputState.concat(getWeightMatrixText(neuralNetwork, 1));
        return outputState;
    }

    public static String getNeuralNetworkAnswer(NeuralNetwork neuralNetwork) {
        String outputState = "NeuralNetworkAnswer() rigth answer".concat(System.lineSeparator());
        for (int layerIndex = 0; layerIndex < neuralNetwork.getRightAnswer().length; layerIndex++) {
            outputState = String.join(" ", outputState, String.format("%2.10f", neuralNetwork.getRightAnswer()[layerIndex]));
        }
        outputState = outputState.concat(System.lineSeparator()).concat("NeuralNetworkAnswer() before formatted").concat(System.lineSeparator());
        double[] answer = neuralNetwork.getLastLayer().getOutputVector();
        for (int layerIndex = 0; layerIndex < answer.length; layerIndex++) {
            outputState = String.join(" ", outputState, String.format("%2.10f", answer[layerIndex]));
        }
        outputState = outputState.concat(System.lineSeparator()).concat("NeuralNetworkAnswer() after formatted").concat(System.lineSeparator());
        answer = neuralNetwork.getNeuralNetworkAnswerNormalized();
        for (int layerIndex = 0; layerIndex < answer.length; layerIndex++) {
            outputState = String.join(" ", outputState, String.format("%2.10f", answer[layerIndex]));
        }
        return outputState;
    }
}