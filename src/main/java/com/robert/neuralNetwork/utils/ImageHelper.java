package com.robert.neuralNetwork.utils;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

/**
 * Created by RLUKAS on 29.10.2016.
 */
public class ImageHelper {

    public static void createImage(double[] array, String imagename) throws IOException {
        BufferedImage b = new BufferedImage(28, 28, BufferedImage.TYPE_INT_RGB);
        for(int i = 0; i < 28; i++) {
            for(int j = 0; j < 28; j++) {
                int rgb = (int)array[(i * 28) + j]<<16 | (int)array[(i * 28) + j] << 8 | (int)array[(i * 28) + j];
                b.setRGB(i, j, rgb);
            }
        }
        ImageIO.write(b, "PNG", new File("C:\\RLukas\\Projekty\\IntelijIdea\\Lista2\\src\\main\\resources\\output\\"+imagename));
    }

    public static void createImage2(double[] array, String imagename) throws IOException {
        BufferedImage image = new BufferedImage(28, 28, BufferedImage.TYPE_3BYTE_BGR);
        int[] data = new int[array.length];
        for (int i = 0; i < data.length; i++) {
            data[i] = (int)array[i];
        }

        image.setRGB(0, 0, 28, 28, data, 0, 28);
        ImageIO.write(image, "PNG", new File("C:\\RLukas\\Projekty\\IntelijIdea\\Lista2\\src\\main\\resources\\output\\"+imagename));
    }

    public static void createImage3(double[] array, String imagename) throws IOException {
        BufferedImage image = new BufferedImage(28, 28, BufferedImage.TYPE_INT_ARGB);
        int[] data = new int[array.length];
        for (int i = 0; i < data.length; i++) {
            int gray = (int) (255 - array[i]);
            data[i] = 0xFF000000 | (gray<<16) | (gray<<8) | gray;;
        }

        image.setRGB(0, 0, 28, 28, data, 0, 28);
        ImageIO.write(image, "PNG", new File("C:\\RLukas\\Projekty\\IntelijIdea\\Lista2\\src\\main\\resources\\output\\"+imagename));
    }
}
