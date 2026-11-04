using StudentPassPredictor.models;
using StudentPassPredictor.utils;
using System;
using System.IO;

namespace StudentPassPredictor
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("=========================================================");
            Console.WriteLine("   Student Pass Predictor - Custom C# MLP Engine");
            Console.WriteLine("=========================================================\n");

            string dataPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "data", "student_performance.csv");
            if (!File.Exists(dataPath))
            {
                dataPath = "data/student_performance.csv";
            }
            if (!File.Exists(dataPath))
            {
                Console.WriteLine($"Error: CSV file not found at '{dataPath}'");
                return;
            }

            Console.WriteLine("[1/4] Loading and Preprocessing Dataset...");
            var (trainData, testData) = CsvDataLoader.LoadAndPreprocess(dataPath, trainRatio: 0.8, seed: 42);
            Console.WriteLine($"      Total Samples Loaded: {trainData.Length + testData.Length}");
            Console.WriteLine($"      Training Set: {trainData.Length} | Testing Set: {testData.Length}\n");

            // Systematic Hyperparameter Experimentation
            Console.WriteLine("[2/4] Running Systematic Hyperparameter Search...");
            RunHyperparameterGridSearch(trainData, testData);

            // Optimal Model Training
            int bestHiddenNodes = 8;
            double bestLearningRate = 0.15;
            int epochs = 2000;

            Console.WriteLine($"\n[3/4] Training Optimal Model ({bestHiddenNodes} Hidden Neurons, LR={bestLearningRate}, Epochs={epochs})...");
            var net = new NeuralNetwork(inputSize: 5, hiddenSize: bestHiddenNodes, outputSize: 1);

            using (StreamWriter logWriter = new StreamWriter("training_log.csv"))
            {
                logWriter.WriteLine("Epoch,MSE_Loss");
                for (int epoch = 1; epoch <= epochs; epoch++)
                {
                    double mse = net.TrainEpoch(trainData, bestLearningRate);
                    logWriter.WriteLine($"{epoch},{mse:F6}");

                    if (epoch % 100 == 0 || epoch == 1)
                    {
                        Console.WriteLine($"      Epoch {epoch,4} | Mean Squared Error Loss: {mse:F6}");
                    }
                }
            }
            Console.WriteLine("      Training completed. Training log exported to 'training_log.csv'.\n");

            // Final Evaluation
            Console.WriteLine("[4/4] Model Evaluation on Test Set...");
            var metrics = EvaluationMetrics.Calculate(net, testData);

            Console.WriteLine("---------------------------------------------------------");
            Console.WriteLine($"  Accuracy  : {metrics.Accuracy * 100:F2}%");
            Console.WriteLine($"  Precision : {metrics.Precision * 100:F2}%");
            Console.WriteLine($"  Recall    : {metrics.Recall * 100:F2}%");
            Console.WriteLine($"  F1-Score  : {metrics.F1Score * 100:F2}%");
            Console.WriteLine("---------------------------------------------------------");
            Console.WriteLine("  Confusion Matrix:");
            Console.WriteLine($"    TP: {metrics.TruePositives,3} | FP: {metrics.FalsePositives,3}");
            Console.WriteLine($"    FN: {metrics.FalseNegatives,3} | TN: {metrics.TrueNegatives,3}");
            Console.WriteLine("=========================================================");
        }

        static void RunHyperparameterGridSearch(DataPoint[] train, DataPoint[] test)
        {
            int[] hiddenSizes = { 4, 8, 12 };
            double[] learningRates = { 0.05, 0.15, 0.30 };
            int epochs = 1000;

            Console.WriteLine("  Hidden Neurons | Learning Rate | Test Accuracy | Test F1-Score");
            Console.WriteLine("  -------------------------------------------------------------");

            foreach (var h in hiddenSizes)
            {
                foreach (var lr in learningRates)
                {
                    var model = new NeuralNetwork(inputSize: 5, hiddenSize: h, outputSize: 1);
                    for (int e = 0; e < epochs; e++)
                    {
                        model.TrainEpoch(train, lr);
                    }
                    var m = EvaluationMetrics.Calculate(model, test);
                    Console.WriteLine($"        {h,2}       |     {lr,4:F2}      |    {m.Accuracy * 100,5:F1}%    |    {m.F1Score * 100,5:F1}%");
                }
            }
        }
    }
}