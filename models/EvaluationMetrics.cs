using StudentPassPredictor.models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StudentPassPredictor.models
{
    public class EvaluationMetrics
    {
        public double Accuracy { get; set; }
        public double Precision { get; set; }
        public double Recall { get; set; }
        public double F1Score { get; set; }
        public int TruePositives { get; set; }
        public int TrueNegatives { get; set; }
        public int FalsePositives { get; set; }
        public int FalseNegatives { get; set; }

        public static EvaluationMetrics Calculate(NeuralNetwork nn, DataPoint[] testData, double threshold = 0.5)
        {
            int tp = 0, tn = 0, fp = 0, fn = 0;

            foreach (var sample in testData)
            {
                var (_, output) = nn.Forward(sample.Features);
                int predicted = output >= threshold ? 1 : 0;
                int actual = (int)sample.Target;

                if (predicted == 1 && actual == 1) tp++;
                else if (predicted == 0 && actual == 0) tn++;
                else if (predicted == 1 && actual == 0) fp++;
                else if (predicted == 0 && actual == 1) fn++;
            }

            int total = testData.Length;
            double accuracy = (double)(tp + tn) / total;
            double precision = (tp + fp) > 0 ? (double)tp / (tp + fp) : 0.0;
            double recall = (tp + fn) > 0 ? (double)tp / (tp + fn) : 0.0;
            double f1 = (precision + recall) > 0 ? 2 * (precision * recall) / (precision + recall) : 0.0;

            return new EvaluationMetrics
            {
                Accuracy = accuracy,
                Precision = precision,
                Recall = recall,
                F1Score = f1,
                TruePositives = tp,
                TrueNegatives = tn,
                FalsePositives = fp,
                FalseNegatives = fn
            };
        }
    }
}
