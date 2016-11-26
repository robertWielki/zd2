package com.robert.neuralNetwork;


public class Settings {

    public static final double LEARNING_RATE = 0.3;
    public static final int BATCH_SIZE = 500;
    public static final int MAX_EPOCH = 500;
    public static final double MIN_ERROR = 0.13;
    public static final int HIDDEN_INPUTS_LAYER_SIZE = 300;

    public static final String IMAGES_10K_FILE = "dataSet/t10k-images.idx3-ubyte";
    public static final String LABELS_10K_FILE = "dataSet/t10k-labels.idx1-ubyte";
    public static final String IMAGES_TRAINING_FILE = "dataSet/train-images.idx3-ubyte";
    public static final String LABELS_TRAINING_FILE = "dataSet/train-labels.idx1-ubyte";

}
