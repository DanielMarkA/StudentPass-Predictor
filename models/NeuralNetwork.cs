using StudentPassPredictor.models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StudentPassPredictor.models
{
    public class NeuralNetwork
    {
        public int InputSize { get; }
        public int HiddenSize { get; }
        public int OutputSize { get; }

        public double[,] WeightsInputHidden { get; private set; }
        public double[,] WeightsHiddenOutput { get; private set; }
        public double[] HiddenBiases { get; private set; }
        public double[] OutputBiases { get; private set; }

        private readonly Random _random = new Random(42); // Fixed seed for reproducibility

        public NeuralNetwork(int inputSize, int hiddenSize, int outputSize = 1)
        {
            InputSize = inputSize;
            HiddenSize = hiddenSize;
            OutputSize = outputSize;

            WeightsInputHidden = new double[InputSize, HiddenSize];
            WeightsHiddenOutput = new double[HiddenSize, OutputSize];
            HiddenBiases = new double[HiddenSize];
            OutputBiases = new double[OutputSize];

            InitializeWeights();
        }

        private void InitializeWeights()
        {
            // Xavier / Glorot Initialization
            double limitInput = Math.Sqrt(6.0 / (InputSize + HiddenSize));
            for (int i = 0; i < InputSize; i++)
                for (int j = 0; j < HiddenSize; j++)
                    WeightsInputHidden[i, j] = (_random.NextDouble() * 2 - 1) * limitInput;

            double limitHidden = Math.Sqrt(6.0 / (HiddenSize + OutputSize));
            for (int i = 0; i < HiddenSize; i++)
                for (int j = 0; j < OutputSize; j++)
                    WeightsHiddenOutput[i, j] = (_random.NextDouble() * 2 - 1) * limitHidden;

            for (int j = 0; j < HiddenSize; j++) HiddenBiases[j] = 0.0;
            for (int j = 0; j < OutputSize; j++) OutputBiases[j] = 0.0;
        }

        private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));
        private static double SigmoidDerivative(double output) => output * (1.0 - output);

        public (double[] HiddenOutputs, double Output) Forward(double[] inputs)
        {
            double[] hiddenOutputs = new double[HiddenSize];
            for (int j = 0; j < HiddenSize; j++)
            {
                double sum = HiddenBiases[j];
                for (int i = 0; i < InputSize; i++)
                    sum += inputs[i] * WeightsInputHidden[i, j];
                hiddenOutputs[j] = Sigmoid(sum);
            }

            double outputSum = OutputBiases[0];
            for (int j = 0; j < HiddenSize; j++)
                outputSum += hiddenOutputs[j] * WeightsHiddenOutput[j, 0];

            double finalOutput = Sigmoid(outputSum);
            return (hiddenOutputs, finalOutput);
        }

        public double TrainEpoch(DataPoint[] trainData, double learningRate)
        {
            double totalError = 0.0;

            foreach (var sample in trainData)
            {
                // 1. Forward Pass
                var (hiddenOutputs, output) = Forward(sample.Features);

                // 2. Compute Loss (MSE)
                double target = sample.Target;
                double error = target - output;
                totalError += error * error;

                // 3. Backpropagation Output Layer
                double outputDelta = error * SigmoidDerivative(output);

                // 4. Backpropagation Hidden Layer
                double[] hiddenDeltas = new double[HiddenSize];
                for (int j = 0; j < HiddenSize; j++)
                {
                    double hiddenError = outputDelta * WeightsHiddenOutput[j, 0];
                    hiddenDeltas[j] = hiddenError * SigmoidDerivative(hiddenOutputs[j]);
                }

                // 5. Update Weights & Biases
                for (int j = 0; j < HiddenSize; j++)
                {
                    WeightsHiddenOutput[j, 0] += learningRate * outputDelta * hiddenOutputs[j];
                }
                OutputBiases[0] += learningRate * outputDelta;

                for (int i = 0; i < InputSize; i++)
                {
                    for (int j = 0; j < HiddenSize; j++)
                    {
                        WeightsInputHidden[i, j] += learningRate * hiddenDeltas[j] * sample.Features[i];
                    }
                }

                for (int j = 0; j < HiddenSize; j++)
                {
                    HiddenBiases[j] += learningRate * hiddenDeltas[j];
                }
            }

            return totalError / trainData.Length;
        }
    }
}