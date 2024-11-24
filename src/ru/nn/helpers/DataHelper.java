package ru.nn.helpers;

import ru.nn.mainclasses.NNLayer;
import ru.nn.mainclasses.NeuralNetwork;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class DataHelper {
    private final static String EURUSD60_csv = "EURUSD60.csv";
    private final static String GBPUSD60_csv = "GBPUSD60.csv";
    private final static String WEIGHT_MATRIX_FILE_NAME = "weight_matrix_";
    private final static String NN_NAME = "nn_";
    private final static String UNDERLINE_SYMBOL = "_";
    private final static int INPUT_UNITS_SIZE = 480;
    private final static int OUTPUT_NEURAL_NETWORK_VECTOR_SIZE = 48;
    private final static String EMPTY_STRING = "";
    private String filePath;
    private String filePathEurUsd;
    private String filePathGbpUsd;
    private List<String> linesEurUsd;
    private List<String> linesGbpUsd;
    private List<DataUnit> dataUnitList;
    private int unitArraySize;

    public DataHelper(String filePath) {
        this.filePath = filePath;
        this.filePathEurUsd = this.filePath.concat(EURUSD60_csv);
        this.filePathGbpUsd = this.filePath.concat(GBPUSD60_csv);
        this.dataUnitList = new ArrayList<DataUnit>();
    }

    public void loadData() {
        try {
            linesEurUsd = Files.lines(Paths.get(filePathEurUsd), StandardCharsets.UTF_8).toList();
            linesGbpUsd = Files.lines(Paths.get(filePathGbpUsd), StandardCharsets.UTF_8).toList();

            for (int index = 0; index < linesEurUsd.size(); index++) {
                dataUnitList.add(new DataUnit(linesEurUsd.get(index)));
                /*
                String lineGbp = findSameRecord(linesGbpUsd, linesEurUsd.get(index));
                if (lineGbp != null){
                    dataUnitList.add(new DataUnit(linesEurUsd.get(index), lineGbp));
                }

                 */
            }
        } catch (IOException exception) {

        }
    }

    public void saveWeightMatrixes(NeuralNetwork neuralNetwork) {
        String nnParameterText = getNeuralNetworkParameterText(neuralNetwork);
        for (int layerIndex = 1; layerIndex < neuralNetwork.getLayersSize(); layerIndex++) {
            Path path = Paths.get(this.filePath.concat(WEIGHT_MATRIX_FILE_NAME).concat(nnParameterText).concat(neuralNetwork.getNnLayersList().get(layerIndex).getLayerName()).concat(".txt"));
            try {
                Files.deleteIfExists(path);
                Files.createFile(path);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }

            NNLayer layer = neuralNetwork.getNnLayersList().get(layerIndex);
            for (int weightIndexRow = 0; weightIndexRow < layer.getInputWeihtsCountPerNode(); weightIndexRow++) {
                List<String> lineDouble = new ArrayList<String>();
                for (int weightIndexColumn = 0; weightIndexColumn < layer.getNodesCount(); weightIndexColumn++) {
                    lineDouble.add(String.valueOf(layer.getWeightMatrix()[weightIndexRow][weightIndexColumn]));
                }
                try {
                    Files.writeString(path, String.join(",", lineDouble).concat(System.lineSeparator()), StandardCharsets.UTF_8, StandardOpenOption.APPEND);
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
        }
    }

    public boolean loadWeightMatrixes(NeuralNetwork neuralNetwork) {
        String nnParameterText = getNeuralNetworkParameterText(neuralNetwork);
        for (int layerIndex = 1; layerIndex < neuralNetwork.getLayersSize(); layerIndex++) {
            Path path = Paths.get(this.filePath.concat(WEIGHT_MATRIX_FILE_NAME).concat(nnParameterText).concat(neuralNetwork.getNnLayersList().get(layerIndex).getLayerName()).concat(".txt"));
            if (!Files.exists(path)) {
                return false;
            }
            try {
                List<String> lines = Files.lines(path, StandardCharsets.UTF_8).toList();
                for (int index = 0; index < lines.size(); index++) {
                    String[] weightRow = lines.get(index).split(",");
                    int getInputWeihtsCountPerNode = neuralNetwork.getNnLayersList().get(layerIndex).getNodesCount();
                    if (getInputWeihtsCountPerNode != weightRow.length) {
                        throw new IOException();
                    }
                    double[] weightRowDouble = Arrays.stream(weightRow).mapToDouble(Double::parseDouble).toArray();
                    neuralNetwork.getNnLayersList().get(layerIndex).getWeightMatrix()[index] = weightRowDouble;
                }
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        return true;
    }

    public String getNeuralNetworkParameterText(NeuralNetwork neuralNetwork) {
        String result = NN_NAME;
        for (int layerIndex = 0; layerIndex < neuralNetwork.getLayersSize(); layerIndex++) {
            result = result.concat(String.valueOf(neuralNetwork.getNnLayersList().get(layerIndex).getNodesCount())).concat(UNDERLINE_SYMBOL);
        }
        return result;
    }

    private String findSameRecord(List<String> pricesList, String targetRecord) {
        String[] targetRecordParts = targetRecord.split(",");
        for (int recordIndex = 0; recordIndex < pricesList.size(); recordIndex++) {
            String[] priceParts = pricesList.get(recordIndex).split(",");
            if (priceParts[0].equals(targetRecordParts[0]) && priceParts[1].equals(targetRecordParts[1])) {
                return pricesList.get(recordIndex);
            }
        }
        return null;
    }

    public List<DataUnit> getPriseList() {
        return this.dataUnitList;
    }

    public double[] getDataUnitArrayByIndex(int dataUnitIndex) {
        List<Double> outputArray = new ArrayList<Double>();//double[unitArraySize*BATCH_SIZE];
        for (int i = dataUnitIndex; i < (INPUT_UNITS_SIZE + dataUnitIndex); i++) {
            double[] units = this.dataUnitList.get(i).getUnitAsArray();
            outputArray.addAll(Arrays.stream(units).boxed().toList());
        }
        double[] out = new double[outputArray.size()];
        for (int i = 0; i < outputArray.size(); i++) {
            out[i] = outputArray.get(i);
        }
        return out;
    }

    public double[] getRightAnswer(int dataUnitIndex) {
        double[] out = new double[OUTPUT_NEURAL_NETWORK_VECTOR_SIZE];
        double lastUnitClosePoint = this.dataUnitList.get(INPUT_UNITS_SIZE + dataUnitIndex - 1).usdClose;
        int outputIndex = 0;
        for (int dataUnitListIndex = 0; dataUnitListIndex < OUTPUT_NEURAL_NETWORK_VECTOR_SIZE / 2; dataUnitListIndex++) {
            out[outputIndex] = lastUnitClosePoint - this.dataUnitList.get(INPUT_UNITS_SIZE + dataUnitIndex + dataUnitListIndex).usdMax;
            outputIndex++;
            out[outputIndex] = lastUnitClosePoint - this.dataUnitList.get(INPUT_UNITS_SIZE + dataUnitIndex + dataUnitListIndex).usdMin;
            outputIndex++;
        }
        for (int outputIndexPostProcess = 0; outputIndexPostProcess < OUTPUT_NEURAL_NETWORK_VECTOR_SIZE; outputIndexPostProcess++) {
            out[outputIndexPostProcess] = valueConverterSigmaXToY(out[outputIndexPostProcess]);
        }
        return out;
    }

    public String getRightAnswerTextLine(int dataUnitIndex) {
        String outputState = "NeuralNetworkAnswer() rigth answer".concat(System.lineSeparator());
        for (int layerIndex = 0; layerIndex < getRightAnswer(dataUnitIndex).length; layerIndex++) {
            outputState = String.join(" ", outputState, String.format("%2.10f", getRightAnswer(dataUnitIndex)[layerIndex]));
        }
        return outputState;
    }

    public int getInputUnitsSize() {
        return INPUT_UNITS_SIZE;
    }

    public int getSingleUnitSize() {
        return this.dataUnitList.get(0).getUnitSize();
    }

    class DataUnit {
        DateTimeFormatter formatter;
        LocalDateTime dateTime;
        private String[] firstSourceLineText;
        private String[] secondSourceLineText;
        private double usdOpen;
        private double usdMax;
        private double usdMin;
        private double usdClose;
        private double usdVolume;
        private double gbpOpen;
        private double gbpMax;
        private double gbpMin;
        private double gbpClose;
        private double gbpVolume;
        private double hour;
        private double dayOfWeek;
        private double dayOfMonth;
        private double month;
        private double[] unitArray;

        public DataUnit(String firstSource, String secondSource) {
            this.firstSourceLineText = firstSource.split(",");
            this.secondSourceLineText = secondSource.split(",");
            this.formatter = DateTimeFormatter.ofPattern("yyyy.MM.dd HH:mm");
            this.dateTime = LocalDateTime.parse(this.firstSourceLineText[0].concat(" ").concat(this.firstSourceLineText[1]), this.formatter);
            this.hour = this.dateTime.getHour();
            this.dayOfWeek = this.dateTime.getDayOfWeek().getValue();
            this.dayOfMonth = this.dateTime.getDayOfMonth();
            this.month = this.dateTime.getMonthValue();
            this.gbpOpen = Double.parseDouble(this.secondSourceLineText[2]);
            this.gbpMax = Double.parseDouble(this.secondSourceLineText[3]);
            this.gbpMin = Double.parseDouble(this.secondSourceLineText[4]);
            this.gbpClose = Double.parseDouble(this.secondSourceLineText[5]);
            this.gbpVolume = Double.parseDouble(this.secondSourceLineText[6]);
            this.usdOpen = Double.parseDouble(this.firstSourceLineText[2]);
            this.usdMax = Double.parseDouble(this.firstSourceLineText[3]);
            this.usdMin = Double.parseDouble(this.firstSourceLineText[4]);
            this.usdClose = Double.parseDouble(this.firstSourceLineText[5]);
            this.usdVolume = Double.parseDouble(this.firstSourceLineText[6]);
            this.unitArray = new double[]{this.gbpOpen, this.gbpMax, this.gbpMin, this.gbpClose, this.gbpVolume, this.usdOpen, this.usdMax, this.usdMin, this.usdClose, this.usdVolume, this.hour, this.dayOfWeek, this.dayOfMonth, this.month};
            unitArraySize = this.unitArray.length;
        }

        public DataUnit(String firstSource) {
            this.firstSourceLineText = firstSource.split(",");
            this.formatter = DateTimeFormatter.ofPattern("yyyy.MM.dd HH:mm");
            this.dateTime = LocalDateTime.parse(this.firstSourceLineText[0].concat(" ").concat(this.firstSourceLineText[1]), this.formatter);
            this.hour = this.dateTime.getHour();
            this.dayOfWeek = this.dateTime.getDayOfWeek().getValue();
            this.dayOfMonth = this.dateTime.getDayOfMonth();
            this.month = this.dateTime.getMonthValue();
            this.usdOpen = Double.parseDouble(this.firstSourceLineText[2]);
            this.usdMax = Double.parseDouble(this.firstSourceLineText[3]);
            this.usdMin = Double.parseDouble(this.firstSourceLineText[4]);
            this.usdClose = Double.parseDouble(this.firstSourceLineText[5]);
            this.usdVolume = 1.0; //Double.parseDouble(this.firstSourceLineText[6]);
            this.unitArray = new double[]{this.usdOpen, this.usdMax, this.usdMin, this.usdClose, this.usdVolume, this.hour, this.dayOfWeek, this.dayOfMonth, this.month};
            unitArraySize = this.unitArray.length;
        }

        public double[] getUnitAsArray() {
            return this.unitArray;
        }

        public int getUnitSize() {
            return this.unitArray.length;
        }
    }

    public double valueConverterSigmaXToY(double x) {
        return 1.0 / (1.0 + Math.exp(-1 * x));
    }
}
