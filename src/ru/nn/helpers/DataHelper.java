package ru.nn.helpers;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class DataHelper {
    private final static int INPUT_UNITS_SIZE = 48;
    private final static int OUTPUT_NEURAL_NETWORK_VECTOR_SIZE = 8;
    private String filePath;
    private List<String> lines;
    private List<DataUnit> dataUnitList;
    private int unitArraySize;

    public DataHelper(String filePath) {
        this.filePath = filePath;
        this.dataUnitList = new ArrayList<DataUnit>();
    }

    public void loadData() {
        try {
            final Integer lineNumber = 0;
            lines = Files.lines(Paths.get(filePath), StandardCharsets.UTF_8).toList();
            String line;
            for (int index = 0; index < lines.size(); index++) {
                dataUnitList.add(new DataUnit(lines.get(index)));
            }
        } catch (IOException exception) {

        }
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
        double[] out = new double[8];
        double lastUnitClosePoint = this.dataUnitList.get(INPUT_UNITS_SIZE + dataUnitIndex - 1).close;
        int outputIndex = 0;
        for (int dataUnitListIndex = 0; dataUnitListIndex < OUTPUT_NEURAL_NETWORK_VECTOR_SIZE / 2; dataUnitListIndex++) {
            out[outputIndex] = lastUnitClosePoint - this.dataUnitList.get(INPUT_UNITS_SIZE + dataUnitIndex).max;
            outputIndex++;
            out[outputIndex] = lastUnitClosePoint - this.dataUnitList.get(INPUT_UNITS_SIZE + dataUnitIndex).min;
            outputIndex++;
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
        private String[] lineText;
        private double open;
        private double max;
        private double min;
        private double close;
        private double volume;
        private double hour;
        private double dayOfWeek;
        private double dayOfMonth;
        private double month;
        private double[] unitArray;

        public DataUnit(String lineText) {
            this.lineText = lineText.split(",");
            this.formatter = DateTimeFormatter.ofPattern("yyyy.MM.dd HH:mm");
            this.dateTime = LocalDateTime.parse(this.lineText[0].concat(" ").concat(this.lineText[1]), this.formatter);
            this.hour = this.dateTime.getHour();
            this.dayOfWeek = this.dateTime.getDayOfWeek().getValue();
            this.dayOfMonth = this.dateTime.getDayOfMonth();
            this.month = this.dateTime.getMonthValue();
            this.open = Double.parseDouble(this.lineText[2]);
            this.max = Double.parseDouble(this.lineText[3]);
            this.min = Double.parseDouble(this.lineText[4]);
            this.close = Double.parseDouble(this.lineText[5]);
            this.volume = Double.parseDouble(this.lineText[6]);
            this.unitArray = new double[]{this.open, this.max, this.min, this.close, this.volume, this.hour, this.dayOfWeek, this.dayOfMonth, this.month};
            unitArraySize = this.unitArray.length;
        }

        public double[] getUnitAsArray() {
            return this.unitArray;
        }

        public int getUnitSize() {
            return this.unitArray.length;
        }
    }
}
