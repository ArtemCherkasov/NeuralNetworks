import ru.nn.helpers.DataHelper;
import ru.nn.helpers.OutputTextHelper;
import ru.nn.mainclasses.NeuralNetwork;

import java.util.Date;

public class Main {
    private final static String PATH_TO_DATA_DIR = "\\data\\";
    private final static int INPUT_SIZE = 10;
    private final static int OUTPUT_SIZE = 10;
    private final static int[] NN_LAYERS = {18, 36, 36, 18, 4};

    public static void main(String[] args) {
        // TODO Auto-generated method stub
        String filePath = System.getProperty("user.dir").concat(PATH_TO_DATA_DIR);
        System.out.println(filePath);
        DataHelper dataHelper = new DataHelper(filePath);
        dataHelper.loadData();

        Date date = new Date();
        System.out.println("Neural networks start" + date);
        int[] nnLayers = {0, 0, 0, 0};
        nnLayers[0] = dataHelper.getDataUnitArrayByIndex(0).length;
        nnLayers[1] = dataHelper.getDataUnitArrayByIndex(0).length * 2;
        nnLayers[2] = dataHelper.getRightAnswer(0).length;
        nnLayers[nnLayers.length - 1] = dataHelper.getRightAnswer(0).length;
        NeuralNetwork neuralNetwork = new NeuralNetwork(nnLayers);
        neuralNetwork.setTechnicalParameters(dataHelper);

        if (!neuralNetwork.loadWeightMatrixes(dataHelper)) {
            for (int batchIndex = 0; batchIndex < 3; batchIndex++) {
                for (int i = 0; i < 60000; i++) {
                    //System.out.println("batch " + batchIndex + " " + dataHelper.getRightAnswerTextLine(i));
                    neuralNetwork.setInputVector(dataHelper.getDataUnitArrayByIndex(i));
                    neuralNetwork.setRightAnswer(dataHelper.getRightAnswer(i));
                    neuralNetwork.getForwardPropagation().step();
                    neuralNetwork.getBackPropagation().step();
                    System.out.println("batch " + batchIndex + " " + neuralNetwork.getAverageQuadraticErrorTextLine());
                }
            }
            neuralNetwork.saveWeightMatrixes(dataHelper);
        } else {

        }
        //neuralNetwork.setRightAnswer(new double[]{0.0, 0.0, 0.0, 0.0});
        double[] vectorToInput = dataHelper.getDataUnitArrayByIndex(0);
        neuralNetwork.setInputVector(vectorToInput);
        neuralNetwork.getForwardPropagation().step();
        //System.out.println(OutputTextHelper.getStateText(neuralNetwork));
        System.out.println(OutputTextHelper.getNeuralNetworkAnswer(neuralNetwork));
        //System.out.println(OutputTextHelper.getBackPropagationInfo(neuralNetwork));

        System.out.println("__________________________________________________________");
    }

}