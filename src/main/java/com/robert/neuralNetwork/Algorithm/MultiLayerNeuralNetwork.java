package com.robert.neuralNetwork.Algorithm;

import com.robert.neuralNetwork.mnist.MyImage;
import org.jblas.DoubleMatrix;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class MultiLayerNeuralNetwork {

    private double learningRate;
    private int maxEpoch;
    private int batchSize;
    private final double minError;
    private DoubleMatrix hiddenWeights;
    private DoubleMatrix outputWeights;
    private DoubleMatrix inputValues;
    private DoubleMatrix outputValues;
    private int hiddenLayerSize;
    private int inputSize;
    private int outputSize;
    private int trainingSize;

    private MultiLayerNeuralNetwork(double learningRate, int maxEpoch, int batchSize,
                                    double minError, DoubleMatrix inputValues, DoubleMatrix outputValues,
                                    int trainingSize, int hiddenLayerSize) {

        this.learningRate = learningRate;
        this.maxEpoch = maxEpoch;
        this.batchSize = batchSize;
        this.minError = minError;
        this.inputValues = inputValues.divi(255.0);
        this.outputValues = outputValues;
        this.trainingSize = trainingSize;
        this.hiddenLayerSize = hiddenLayerSize;
        inputSize = inputValues.getColumns();
        outputSize = 10; //TODO variable
        initializeWeights();
    }

    private void initializeWeights() {
        hiddenWeights = DoubleMatrix.rand(hiddenLayerSize, inputSize).div(inputSize);
        outputWeights = DoubleMatrix.rand(outputSize, hiddenLayerSize).div(hiddenLayerSize);
    }

    public List train() {
        Random random = new Random();
        int[] nn = new int[batchSize];
        double error = Double.MAX_VALUE;
        List<Double> errors = new ArrayList<>(maxEpoch/4);

        for (int epoch = 0; epoch < maxEpoch && error > minError; epoch++) {

            for (int i = 0; i < batchSize; i++) {

                nn[i] = random.nextInt(trainingSize);

                DoubleMatrix inputVector = inputValues.getRow(nn[i]).transpose();
                DoubleMatrix hiddenActualInput = hiddenWeights.mmul(inputVector);
                DoubleMatrix hiddenOutputVector = activationFunctionOnMatrix(hiddenActualInput);
                DoubleMatrix outputActualInput = outputWeights.mmul(hiddenOutputVector);
                DoubleMatrix outputVector = activationFunctionOnMatrix(outputActualInput);

                DoubleMatrix targetVector = createVectorFromOutputValues((int) outputValues.get(nn[i], 0));

                DoubleMatrix outputDelta = dActivationFunctionOnMatrix(outputActualInput).mul(outputVector.sub(targetVector));
                DoubleMatrix hiddenDelta = dActivationFunctionOnMatrix(hiddenActualInput).mul(outputWeights.transpose().mmul(outputDelta));

                outputWeights = outputWeights.sub(outputDelta.mmul(hiddenOutputVector.transpose()).muli(learningRate));
                hiddenWeights = hiddenWeights.sub(hiddenDelta.mmul(inputVector.transpose()).muli(learningRate));
            }
            error = 0;
            for (int i = 0; i < batchSize; i++) {
                DoubleMatrix inputVector = inputValues.getRow(nn[i]).transpose();
                DoubleMatrix targetVector = createVectorFromOutputValues((int) outputValues.get(nn[i], 0));
                error += activationFunctionOnMatrix(outputWeights.mmul(
                        activationFunctionOnMatrix(hiddenWeights.mmul(inputVector)))).sub(targetVector).norm2();
            }
            error = error/batchSize;
            errors.add(error);
            System.out.println(epoch + ")\terrors\t" + error);
        }
        return errors;
    }

    private DoubleMatrix activationFunctionOnMatrix(DoubleMatrix matrix) {
        double[] vector = matrix.getColumn(0).toArray();
        for (int i = 0; i < vector.length; i++) {
            vector[i] = logisticSigmoid(vector[i]);
        }
        return new DoubleMatrix(vector);
    }

    private DoubleMatrix dActivationFunctionOnMatrix(DoubleMatrix matrix) {
        double[] vector = matrix.getColumn(0).toArray();
        for (int i = 0; i < vector.length; i++) {
            vector[i] = dLogisticSigmoid(vector[i]);
        }
        return new DoubleMatrix(vector);
    }

    private DoubleMatrix createVectorFromOutputValues(int number) {
        double[] data = new double[outputSize];
        data[number] = 1;
        return new DoubleMatrix(data);
    }

    private double logisticSigmoid(double x){
        return 1. / (1 + Math.exp(-x));
    }

    private double dLogisticSigmoid(double x){
        return logisticSigmoid(x) * (1. - logisticSigmoid(x));
    }

    public void test(List<MyImage> testData) {
        DoubleMatrix images;
        DoubleMatrix labels;

        double[][] matrixData = new double[testData.size()][];
        double[][] matrixData2 = new double[testData.size()][];
        for (int i = 0; i < testData.size(); i++) {
            matrixData[i]  = testData.get(i).getImage();
            matrixData2[i] = new double[]{testData.get(i).getLabel()};
        }
        images = new DoubleMatrix(matrixData).divi(255.0);
        labels = new DoubleMatrix(matrixData2);

        int classificationErrors = 0;
        int correctlyClassified = 0;

        for (int i = 0; i <images.getRows(); i++) {
            DoubleMatrix inputVector = images.getRow(i).transpose();
            DoubleMatrix outputVector = evaluate(inputVector);

            int digit = maxValue(outputVector.getColumn(0).toArray());

            if (digit == (int) labels.get(i, 0)){
                correctlyClassified++;
            }
            else {
                classificationErrors++;
//                ImageHelper.createImage3(inputValue.data, digit+"___"+i+"___" +(int) outputValue.get(i, 0)+".png");
            }
        }
        System.out.println("Classification errors:\t" + classificationErrors);
        System.out.println("Correctly classified: \t" + correctlyClassified);
    }

    private DoubleMatrix evaluate(DoubleMatrix inputVector){
        return activationFunctionOnMatrix(outputWeights.mmul(activationFunctionOnMatrix(hiddenWeights.mmul(inputVector))));
    }

    private int maxValue(double[] array){
        int index = -1;
        double max = Double.MIN_VALUE;
        for (int i = 0; i < array.length; i++) {
            if (max < array[i]) {
                max = array[i];
                index = i;
            }
        }
        return  index;
    }

    @Override
    public String toString() {
        return "MultiLayerNeuralNetwork{" +
                "learningRate=" + learningRate +
                ", maxEpoch=" + maxEpoch +
                ", batchSize=" + batchSize +
                ", minError=" + minError +
                ", hiddenLayerSize=" + hiddenLayerSize +
                ", inputSize=" + inputSize +
                ", outputSize=" + outputSize +
                ", trainingSize=" + trainingSize +
                '}';
    }

    public static class MultiLayerNeuralNetworkBuilder{

        private double learningRate;
        private int maxEpoch;
        private int batchSize;
        private double minError;
        private DoubleMatrix inputValues;
        private DoubleMatrix outputValues;
        private int layerSize;
        private int trainingSize;


        public MultiLayerNeuralNetworkBuilder() {
        }

        public MultiLayerNeuralNetworkBuilder numberOfEpochs(int number){
            this.maxEpoch = number;
            return this;
        }

        public MultiLayerNeuralNetworkBuilder learningRate(double rate){
            this.learningRate = rate;
            return this;
        }

        public MultiLayerNeuralNetworkBuilder batchSize(int size){
            this.batchSize = size;
            return this;
        }

        public MultiLayerNeuralNetworkBuilder minError(double minError) {
            this.minError = minError;
            return this;
        }

        public MultiLayerNeuralNetworkBuilder hiddenLayerSize(int layerSize) {
            this.layerSize = layerSize;
            return this;
        }

        public MultiLayerNeuralNetworkBuilder addTrainingData(List<MyImage> data) {
            trainingSize = data.size();
            double[][] matrixData = new double[data.size()][];
            double[][] matrixData2 = new double[data.size()][];
            for (int i = 0; i < data.size(); i++) {
                matrixData[i]  = data.get(i).getImage();
                matrixData2[i] = new double[]{data.get(i).getLabel()};
            }
            inputValues = new DoubleMatrix(matrixData);
            outputValues = new DoubleMatrix(matrixData2);

            return this;
        }

        public MultiLayerNeuralNetwork build(){
            return new MultiLayerNeuralNetwork(learningRate, maxEpoch, batchSize,
                    minError, inputValues, outputValues, trainingSize, layerSize);
        }

    }

}
