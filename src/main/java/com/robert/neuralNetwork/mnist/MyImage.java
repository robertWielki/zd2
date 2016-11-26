package com.robert.neuralNetwork.mnist;

import lombok.Data;

@Data
public class MyImage {

    private double[] image;
    private int label;

    public MyImage(double[] image, int label) {
        this.image = image;
        this.label = label;
    }
}
