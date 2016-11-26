package com.robert.neuralNetwork;

import com.robert.neuralNetwork.Algorithm.MultiLayerNeuralNetwork;
import com.robert.neuralNetwork.mnist.MNISTLoader;
import com.robert.neuralNetwork.mnist.MyImage;
import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.stage.Stage;

import java.io.IOException;
import java.net.URISyntaxException;
import java.util.List;

public class WithoutGui extends Application {

    @Override
    public void start(Stage primaryStage) throws Exception {
        List<MyImage> trainingData = MNISTLoader.loadMNISTImagesData(Settings.LABELS_TRAINING_FILE, Settings.IMAGES_TRAINING_FILE, false);

        MultiLayerNeuralNetwork network = new MultiLayerNeuralNetwork.MultiLayerNeuralNetworkBuilder()
                .learningRate(Settings.LEARNING_RATE)
                .batchSize(Settings.BATCH_SIZE)
                .numberOfEpochs(Settings.MAX_EPOCH)
                .minError(Settings.MIN_ERROR)
                .hiddenLayerSize(Settings.HIDDEN_INPUTS_LAYER_SIZE)
                .addTrainingData(trainingData)
                .build();

        System.out.println(network.toString());
        showDiagram(network.train(), primaryStage);

        List<MyImage> testData = MNISTLoader.loadMNISTImagesData(Settings.LABELS_10K_FILE, Settings.IMAGES_10K_FILE, false);
        network.test(testData);
    }

    public static void showDiagram(List<Double> errors, Stage stage)  {
        stage.setTitle("Line Chart Sample");
        //defining the axes
        final NumberAxis xAxis = new NumberAxis();
        final NumberAxis yAxis = new NumberAxis();
        xAxis.setLabel("Numer iteracji");
        //creating the chart
        final LineChart<Number,Number> lineChart =
                new LineChart<>(xAxis, yAxis);
        lineChart.setTitle("Wykres błedów");
        //defining a series
        XYChart.Series series = new XYChart.Series();
        series.setName("Błedy");
        for (int i = 0; i < errors.size(); i++) {
            series.getData().add(new XYChart.Data(i, errors.get(i)));
        }

        Scene scene  = new Scene(lineChart,800,600);
        lineChart.getData().add(series);

        stage.setScene(scene);
        stage.show();
    }

    public static void main(String[] args) throws IOException, URISyntaxException {
        launch(args);
    }

}
