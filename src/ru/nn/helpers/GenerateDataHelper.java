package ru.nn.helpers;

import ru.nn.mainclasses.NeuralNetwork;

public class GenerateDataHelper {
    private final static double[] LEARNING_VECTOR = {0.05, 0.10, 0.25, 0.58, 0.05, 0.10, 0.25, 0.58, 0.05, 0.10, 0.25, 0.58, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0};
    private final static double[][] WEIGHT_MATRIX_0 = {{0.15, 0.25}, {0.20, 0.30}};
    private final static double[] BIAS_0 = {0.35, 0.35};
    private final static double[][] WEIGHT_MATRIX_1 = {{0.40, 0.50}, {0.45, 0.55}};
    private final static double[] BIAS_1 = {0.60, 0.60};
    private final static double[] RIGHT_ANSWER = {0.01, 0.99, 0.23, 0.68};
    private final static double FACTOR = 1.0;

    public static double[] getRandomVector(int vectorSize) {
        double[] returnVector = new double[vectorSize];
        for (int i = 0; i < vectorSize; i++) {
            returnVector[i] = Math.random() * FACTOR;
        }
        return returnVector;
    }

    public static double[] getInputVectorTest(int vectorSize) {
        double[] returnVector = new double[vectorSize];
        for (int i = 0; i < vectorSize; i++) {
            returnVector[i] = LEARNING_VECTOR[i];
        }
        return returnVector;
    }

    public static void setWeightMatrixTest(NeuralNetwork neuralNetwork) {
        neuralNetwork.getNnLayersList().get(1).setWeightMatrix(WEIGHT_MATRIX_0);
        neuralNetwork.getNnLayersList().get(0).setWeightBias(BIAS_0);
        neuralNetwork.getNnLayersList().get(2).setWeightMatrix(WEIGHT_MATRIX_1);
        neuralNetwork.getNnLayersList().get(1).setWeightBias(BIAS_1);
    }

    public static double[] getRightAnswer() {
        return RIGHT_ANSWER;
    }

}
