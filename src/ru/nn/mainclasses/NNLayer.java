package ru.nn.mainclasses;

import java.util.Arrays;
import java.util.Random;

public class NNLayer {
    private final static double A_CONSTANT_LINEAR_FUNCTION = 1.0 / 5.0;
    private final static double INITIAL_WEIGHT = 0.25;
    private final static String DEFAULT_LAYER_NAME = "layer without name";
    private final static String LAYER_NAME = "layer ";
    private final static double ACTIVATE_FUNCTION_COEFFICIENT = 1.0;
    private final static double LEARN_SPEED_COEFFICIENT = 0.05;
    private final static double BIAS = 1.0;
    private final static boolean TEST_STATE = true;

    String layerName = DEFAULT_LAYER_NAME;
    int nodesCount;
    int inputWeihtsCountPerNode;
    double[] nodesValueVector;
    double biasValue;
    double[][] weightMatrix;
    double[] outputVector;
    double[] weightBias;
    double[] errorOutputVector;
    double[] dE_dOut;
    double[] dOut_dNet;
    double[][] dNet_dw;
    double[] dNet_dOut;
    double[][] dEtotal_dw;
    double errorTotal;
    double niu;
    boolean isLastLayer = false;
    boolean isFirstLayer = false;
    NNLayer nextLayer;
    NNLayer prevLayer;

    public NNLayer(int inputWeihtsCountPerNode, int nodesCount) {
        // TODO Auto-generated constructor stub
        this.inputWeihtsCountPerNode = inputWeihtsCountPerNode;
        this.nodesCount = nodesCount;
        this.nodesValueVector = new double[this.nodesCount];
        this.weightMatrix = new double[this.inputWeihtsCountPerNode][this.nodesCount];
        this.biasValue = BIAS;
        Random randomWeight = new Random();
        for (int i = 0; i < this.inputWeihtsCountPerNode; i++) {
            for (int j = 0; j < this.nodesCount; j++) {
                this.weightMatrix[i][j] = randomWeight.nextDouble() * 2 - 1;
            }
        }
        /*
        for (int j = 0; j < this.nodesCount; j++) {
            this.weightBias[j] = INITIAL_WEIGHT;
        }
         */
        this.outputVector = new double[this.nodesCount];
        this.errorOutputVector = new double[this.nodesCount];
        this.dE_dOut = new double[this.nodesCount];
        this.dOut_dNet = new double[this.nodesCount];
        this.dNet_dw = new double[this.inputWeihtsCountPerNode][this.nodesCount];
        this.dNet_dOut = new double[this.nodesCount];
        this.dEtotal_dw = new double[this.inputWeihtsCountPerNode][this.nodesCount];
        this.niu = LEARN_SPEED_COEFFICIENT;
    }

    public NNLayer(int inputWeihtsCountPerNode, int nodesCount, String layerNumber) {
        // TODO Auto-generated constructor stub
        this(inputWeihtsCountPerNode, nodesCount);
        this.layerName = LAYER_NAME.concat(layerNumber);
    }

    public NNLayer(int inputWeihtsCountPerNode, int nodesCount, int layerNumber) {
        // TODO Auto-generated constructor stub
        this(inputWeihtsCountPerNode, nodesCount);
        this.layerName = LAYER_NAME.concat(String.valueOf(layerNumber));
    }

    public void setNextLayer(NNLayer nextLayer) {
        this.nextLayer = nextLayer;
        this.weightBias = new double[this.nextLayer.getNodesCount()];
        for (int nextLayerIndex = 0; nextLayerIndex < this.nextLayer.getNodesCount(); nextLayerIndex++) {
            this.weightBias[nextLayerIndex] = INITIAL_WEIGHT;
        }
    }

    public void setPrevLayer(NNLayer prevLayer) {
        this.prevLayer = prevLayer;
    }

    public String getLayerName() {
        return this.layerName;
    }

    public String getLayerNameAndSize() {
        return this.layerName.concat(" ").concat("(").concat(String.valueOf(this.inputWeihtsCountPerNode)).concat(",")
                .concat(String.valueOf(this.nodesCount)).concat(")");
    }

    public void setNodesValueVector(double[] nodesValueVector) {
        this.nodesValueVector = nodesValueVector.clone();
    }

    public double[] getNodesValueVector() {
        return this.nodesValueVector;
    }

    public String getNodesVectorTextline() {
        String outputLine = "input vector, size(" + nodesCount + ")" + System.lineSeparator();
        for (int i = 0; i < nodesCount; i++) {
            outputLine = String.join(" ", outputLine, String.format("%2.2f", nodesValueVector[i]));
        }
        return outputLine;
    }

    public double[][] getWeightMatrix() {
        return this.weightMatrix;
    }

    public void setWeightMatrix(double[][] weightMatrix) {
        this.weightMatrix = weightMatrix;
    }

    public void setWeightBias(double[] weightBias) {
        this.weightBias = weightBias;
    }

    public String getWeightMatrixText() {
        String outputLine = "weight matrix, size(" + inputWeihtsCountPerNode + "," + nodesCount + ")" + System.lineSeparator();
        for (int weightIndex = 0; weightIndex < this.inputWeihtsCountPerNode; weightIndex++) {
            for (int nodeIndex = 0; nodeIndex < this.nodesCount; nodeIndex++) {
                outputLine = String.join(" ", outputLine, String.format("%2.8f", this.weightMatrix[weightIndex][nodeIndex]));
            }
            outputLine = outputLine + System.lineSeparator();
        }
        return outputLine;
    }

    public double[] calculateNodeValueAndOutputVector() {
        if (this.isFirstLayer) {
            this.outputVector = this.nodesValueVector.clone();
            return this.outputVector;
        } else {
            for (int nodeIndex = 0; nodeIndex < this.nodesCount; nodeIndex++) {
                double nodeValue = 0.0;
                for (int weightIndex = 0; weightIndex < this.inputWeihtsCountPerNode; weightIndex++) {
                    nodeValue = nodeValue + this.prevLayer.getOutputVector()[weightIndex] * this.weightMatrix[weightIndex][nodeIndex];
                }
                nodeValue = nodeValue + this.prevLayer.weightBias[nodeIndex] * this.biasValue;
                this.nodesValueVector[nodeIndex] = nodeValue;
                this.outputVector[nodeIndex] = this.activationFunction(nodeValue);
            }
            return this.outputVector;
        }
    }

    public double[] getOutputVector() {
        return this.outputVector;
    }

    public String getOutputVectorTextline() {
        String outputLine = this.layerName + ", output vector, size(" + this.nodesCount + ")" + System.lineSeparator();
        for (int nodeIndex = 0; nodeIndex < this.nodesCount; nodeIndex++) {
            outputLine = String.join(" ", outputLine, String.format("%2.10f", this.outputVector[nodeIndex]));
        }
        return outputLine;
    }

    public double activationFunction(double x) {
        return 1.0 / (1.0 + Math.exp(-1 * x));
        /*
        if (isLastLayer) {
            return A_CONSTANT_LINEAR_FUNCTION * x;
        } else {
            return 1.0 / (1.0 + Math.exp(-1 * x));
        }

         */
    }

    public double derivativeActivationFunction(double x) {
        return x * (1 - x);
        /*
        if (isLastLayer) {
            return A_CONSTANT_LINEAR_FUNCTION;
        } else {
            return x * (1 - x);
        }

         */
    }

    public void calculateErrosVector(double[] rightAnswers) {
        for (int nodeIndex = 0; nodeIndex < this.nodesCount; nodeIndex++) {
            this.errorOutputVector[nodeIndex] = Math.pow(rightAnswers[nodeIndex] - this.outputVector[nodeIndex], 2) / 2.0;
        }
        this.errorTotal = Arrays.stream(this.errorOutputVector).sum();
    }

    public void calculate_dE_dOut_NetworkOut(double[] rightAnswers) {
        for (int nodeIndex = 0; nodeIndex < this.nodesCount; nodeIndex++) {
            this.dE_dOut[nodeIndex] = -1 * (rightAnswers[nodeIndex] - this.outputVector[nodeIndex]);
        }
    }

    public void calculate_dE_dOut() {
        if (!this.isFirstLayer) {
            for (int indexCurrentNode = 0; indexCurrentNode < this.nodesCount; indexCurrentNode++) {
                for (int indexNextLayerNode = 0; indexNextLayerNode < this.nextLayer.getNodesCount(); indexNextLayerNode++) {
                    double nextLayer_getdE_dOut = this.nextLayer.getdE_dOut()[indexNextLayerNode];
                    double nextLayer_getdOut_dNet = this.nextLayer.getdOut_dNet()[indexNextLayerNode];
                    double nextLayer_getWeightMatrix = this.nextLayer.getWeightMatrix()[indexCurrentNode][indexNextLayerNode];
                    double mult = nextLayer_getdE_dOut * nextLayer_getdOut_dNet * nextLayer_getWeightMatrix;
                    this.dE_dOut[indexCurrentNode] = this.dE_dOut[indexCurrentNode] + mult;
                }
            }
        }
    }

    public void calculate_dOut_dNet() {
        for (int nodeIndex = 0; nodeIndex < this.nodesCount; nodeIndex++) {
            this.dOut_dNet[nodeIndex] = derivativeActivationFunction(this.outputVector[nodeIndex]);
        }
    }

    public void calculate_dNet_dw() {
        for (int nodeIndex = 0; nodeIndex < this.nodesCount; nodeIndex++) {
            for (int weightIndex = 0; weightIndex < this.inputWeihtsCountPerNode; weightIndex++) {
                this.dNet_dw[weightIndex][nodeIndex] = this.prevLayer.outputVector[weightIndex];
            }
        }
    }

    public void calculate_dNet_dO() {
        for (int nodeIndex = 0; nodeIndex < this.nodesCount; nodeIndex++) {
            this.dNet_dOut[nodeIndex] = this.nodesValueVector[nodeIndex];
        }
    }

    public void calculate_dEtotal_dw() {
        for (int weightIndex = 0; weightIndex < this.inputWeihtsCountPerNode; weightIndex++) {
            for (int nodeIndex = 0; nodeIndex < this.nodesCount; nodeIndex++) {
                this.dEtotal_dw[weightIndex][nodeIndex] = this.dE_dOut[nodeIndex] * this.dOut_dNet[nodeIndex] * this.dNet_dw[weightIndex][nodeIndex];
            }
        }

    }

    public void correctWeightMatrix() {
        for (int weightIndex = 0; weightIndex < this.inputWeihtsCountPerNode; weightIndex++) {
            for (int nodeIndex = 0; nodeIndex < this.nodesCount; nodeIndex++) {
                double deltaW = weightMatrix[weightIndex][nodeIndex] - this.niu * this.dEtotal_dw[weightIndex][nodeIndex];
                weightMatrix[weightIndex][nodeIndex] = deltaW;
            }
        }
    }

    public String get_dE_dOut_Text() {
        String outputLine = this.layerName + ", dE_dO vector, size(" + this.nodesCount + ")" + System.lineSeparator();
        for (int nodeIndex = 0; nodeIndex < this.nodesCount; nodeIndex++) {
            outputLine = String.join(" ", outputLine, String.format("%2.10f", this.dE_dOut[nodeIndex]));
        }
        return outputLine;
    }

    public String get_dOut_dNet_Text() {
        String outputLine = this.layerName + ", dO_dNet vector, size(" + this.nodesCount + ")" + System.lineSeparator();
        for (int nodeIndex = 0; nodeIndex < this.nodesCount; nodeIndex++) {
            outputLine = String.join(" ", outputLine, String.format("%2.10f", this.dOut_dNet[nodeIndex]));
        }
        return outputLine;
    }

    public String get_dNet_dw_Text() {
        String outputLine = this.layerName + ", dNet_dw vector, size(" + this.nodesCount + ")";
        for (int weightIndex = 0; weightIndex < this.inputWeihtsCountPerNode; weightIndex++) {
            outputLine = outputLine.concat(System.lineSeparator());
            for (int nodeIndex = 0; nodeIndex < this.nodesCount; nodeIndex++) {
                outputLine = String.join(" ", outputLine, String.format("%2.10f", this.dNet_dw[weightIndex][nodeIndex]));
            }
        }

        return outputLine;
    }

    public String get_dEtotal_dw_Text() {
        String outputLine = this.layerName + ", dEtotal_dw vector, size(" + this.nodesCount + ")";
        for (int weightIndex = 0; weightIndex < this.inputWeihtsCountPerNode; weightIndex++) {
            outputLine = outputLine.concat(System.lineSeparator());
            for (int nodeIndex = 0; nodeIndex < this.nodesCount; nodeIndex++) {
                outputLine = String.join(" ", outputLine, String.format("%2.10f", this.dEtotal_dw[weightIndex][nodeIndex]));
            }
        }
        return outputLine;
    }

    public void setErrorOutputVector(double[] errorOutputVector) {
        this.errorOutputVector = errorOutputVector;
    }

    public String getErrorOutputVectorText() {
        //return errorOutputVector;
        String outputLine = this.layerName + ", error output vector, size(" + this.nodesCount + ")" + System.lineSeparator();
        for (int nodeIndex = 0; nodeIndex < this.nodesCount; nodeIndex++) {
            outputLine = String.join(" ", outputLine, String.format("%2.10f", this.errorOutputVector[nodeIndex]));
        }
        return outputLine;
    }

    public double getErrorTotal() {
        return errorTotal;
    }

    public String getErrorTotalText() {
        return String.format("%2.10f", this.errorTotal);
    }

    public int getNodesCount() {
        return nodesCount;
    }

    public void setNodesCount(int nodesCount) {
        this.nodesCount = nodesCount;
    }

    public int getInputWeihtsCountPerNode() {
        return inputWeihtsCountPerNode;
    }

    public void setInputWeihtsCountPerNode(int inputWeihtsCountPerNode) {
        this.inputWeihtsCountPerNode = inputWeihtsCountPerNode;
    }

    public boolean isLastLayer() {
        return isLastLayer;
    }

    public void setLastLayer(boolean isLastLayer) {
        this.isLastLayer = isLastLayer;
    }

    public boolean isFirstLayer() {
        return isFirstLayer;
    }

    public void setFirstLayer(boolean firstLayer) {
        isFirstLayer = firstLayer;
    }

    public double[] getdE_dOut() {
        return dE_dOut;
    }

    public double[] getdOut_dNet() {
        return dOut_dNet;
    }

    public double[][] getdNet_dw() {
        return dNet_dw;
    }

    public double[][] getdEtotal_dw() {
        return dEtotal_dw;
    }
}
