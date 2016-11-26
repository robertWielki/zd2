package com.robert.neuralNetwork.mnist;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.util.ArrayList;
import java.util.List;


public class MNISTLoader {

    public static List<MyImage> loadMNISTImagesData(String labelFileName, String imageFileName,boolean saveAsImageOnDisk) {
        FileInputStream inImage;
        FileInputStream inLabel;
        List<MyImage> result = null;

        int[] hashMap = new int[10];
        try {
            inImage = new FileInputStream(MNISTLoader.class.getClassLoader().getResource(imageFileName).toURI().getPath());
            inLabel =  new FileInputStream(MNISTLoader.class.getClassLoader().getResource(labelFileName).toURI().getPath());

            int magicNumberImages = (inImage.read() << 24) | (inImage.read() << 16) | (inImage.read() << 8) | (inImage.read());
            int numberOfImages = (inImage.read() << 24) | (inImage.read() << 16) | (inImage.read() << 8) | (inImage.read());
            int numberOfRows  = (inImage.read() << 24) | (inImage.read() << 16) | (inImage.read() << 8) | (inImage.read());
            int numberOfColumns = (inImage.read() << 24) | (inImage.read() << 16) | (inImage.read() << 8) | (inImage.read());

            int magicNumberLabels = (inLabel.read() << 24) | (inLabel.read() << 16) | (inLabel.read() << 8) | (inLabel.read());
            int numberOfLabels = (inLabel.read() << 24) | (inLabel.read() << 16) | (inLabel.read() << 8) | (inLabel.read());

            BufferedImage image = new BufferedImage(numberOfColumns, numberOfRows, BufferedImage.TYPE_INT_ARGB);
            int numberOfPixels = numberOfRows * numberOfColumns;
            int[] imgPixels = new int[numberOfPixels];

            result = new ArrayList<>(numberOfImages);

            for(int i = 0; i < numberOfImages; i++) {
                double[] imgPixelsDouble = new double[numberOfPixels];

                if(i % 1000 == 0) {System.out.println("Number of images extracted: " + i);}

                for(int p = 0; p < numberOfPixels; p++) {
                    int gray = 255 - inImage.read();
                    imgPixels[p] = 0xFF000000 | (gray<<16) | (gray<<8) | gray;
                    imgPixelsDouble[p] = 255-gray;

                }
                image.setRGB(0, 0, numberOfColumns, numberOfRows, imgPixels, 0, numberOfColumns);

                int label = inLabel.read();
                hashMap[label]++;

                if (saveAsImageOnDisk){
                    String outputPath = MNISTLoader.class.getClassLoader().getResource("output\\").toURI().getPath();
                    File outputFile = new File(outputPath + + label + "____" + hashMap[label]+  ".png");
                    ImageIO.write(image, "png", outputFile);
                }
                result.add(new MyImage(imgPixelsDouble, label));
            }

        } catch (Exception e) {
            e.printStackTrace();
        }

        return result;
    }

}
