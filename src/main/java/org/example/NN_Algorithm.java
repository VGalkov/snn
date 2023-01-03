package org.example;

import java.util.function.UnaryOperator;

public class NN_Algorithm {

    private final double learningRate;
    private final Layer[] layers;
    private final UnaryOperator<Double> activation;
    private final UnaryOperator<Double> derivative;

    public NN_Algorithm(double learningRate, UnaryOperator<Double> activation, UnaryOperator<Double> derivative, int... sizes) {
        this.learningRate = learningRate;
        this.activation = activation;
        this.derivative = derivative;
        layers = new Layer[sizes.length];

        for (int i = 0; i < sizes.length; i++) {
            int nextSize = sizes[0];
            if(i < sizes.length - 1)
                nextSize = sizes[i + 1];

            layers[i] = new Layer(sizes[i], nextSize);
            for (int j = 0; j < sizes[i]; j++) {
                layers[i].biases[j] = Math.random();
                for (int k = 0; k < nextSize; k++)
                    layers[i].weights[j][k] = Math.random();
            }
        }
    }

    public double[] feedForward(double[] inputs) {
        System.arraycopy(inputs, 0, layers[0].neurons, 0, inputs.length);
        for (int i = 1; i < layers.length; i++)
            for (int j = 0; j < layers[i].size; j++) {
                for (int k = 0; k < layers[i - 1].size; k++)
                    layers[i].neurons[j] += layers[i - 1].neurons[k] * layers[i - 1].weights[k][j];
                layers[i].neurons[j] = activation.apply(layers[i].neurons[j]+layers[i].biases[j]);
            }
        return layers[layers.length - 1].neurons;
    }

    public void backpropagation(double[] targets) {
        double[] errors = new double[layers[layers.length - 1].size];
        for (int i = 0; i < layers[layers.length - 1].size; i++)
            errors[i] = targets[i] - layers[layers.length - 1].neurons[i];

        for (int k = layers.length - 2; k >= 0; k--) {

            double[] gradients = new double[layers[k + 1].size];
            for (int i = 0; i < layers[k + 1].size; i++)
                gradients[i] = learningRate * (errors[i] * derivative.apply(layers[k + 1].neurons[i]));

            double[][] deltas = new double[layers[k + 1].size][layers[k].size];
            for (int i = 0; i < layers[k + 1].size; i++)
                for (int j = 0; j < layers[k].size; j++)
                    deltas[i][j] = gradients[i] * layers[k].neurons[j];

            double[] errorsNext = new double[layers[k].size];
            for (int i = 0; i < layers[k].size; i++)
                for (int j = 0; j < layers[k + 1].size; j++)
                    errorsNext[i] += layers[k].weights[i][j] * errors[j];
            errors = errorsNext.clone();

            double[][] weightsNew = new double[layers[k].weights.length][layers[k].weights[0].length];
            for (int i = 0; i < layers[k + 1].size; i++)
                for (int j = 0; j < layers[k].size; j++)
                    weightsNew[j][i] = layers[k].weights[j][i] + deltas[i][j];

            layers[k].weights = weightsNew;
            for (int i = 0; i < layers[k + 1].size; i++)
                layers[k + 1].biases[i] += gradients[i];
        }
    }

    private  static class Layer {

        public int size;
        public double[] neurons;
        public double[] biases;
        public double[][] weights;

        public Layer(int size, int nextSize) {
            this.size = size;
            neurons = new double[size];
            biases = new double[size];
            weights = new double[size][nextSize];
        }
    }

}