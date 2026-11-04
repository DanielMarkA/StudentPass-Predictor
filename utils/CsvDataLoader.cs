using StudentPassPredictor.models;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StudentPassPredictor.utils
{
    public class CsvDataLoader
    {
        public static (DataPoint[] Train, DataPoint[] Test) LoadAndPreprocess(
            string filePath, double trainRatio = 0.8, int seed = 42)
        {
            var lines = File.ReadAllLines(filePath).Skip(1).Where(l => !string.IsNullOrWhiteSpace(l)).ToList();
            int sampleCount = lines.Count;
            int featureCount = 5;

            double?[,] rawFeatures = new double?[sampleCount, featureCount];
            double[] targets = new double[sampleCount];

            for (int i = 0; i < sampleCount; i++)
            {
                var parts = lines[i].Split(',');
                for (int j = 0; j < featureCount; j++)
                {
                    if (double.TryParse(parts[j], out double val))
                        rawFeatures[i, j] = val;
                    else
                        rawFeatures[i, j] = null; // missing value handling
                }
                targets[i] = double.Parse(parts[5]);
            }

            // 1. Median Imputation
            double[] featureMedians = new double[featureCount];
            for (int j = 0; j < featureCount; j++)
            {
                var vals = new List<double>();
                for (int i = 0; i < sampleCount; i++)
                {
                    if (rawFeatures[i, j].HasValue)
                        vals.Add(rawFeatures[i, j].Value);
                }
                vals.Sort();
                featureMedians[j] = vals.Count % 2 == 1
                    ? vals[vals.Count / 2]
                    : (vals[vals.Count / 2 - 1] + vals[vals.Count / 2]) / 2.0;
            }

            // 2. Impute and Calculate Min / Max
            double[,] cleanFeatures = new double[sampleCount, featureCount];
            double[] mins = Enumerable.Repeat(double.MaxValue, featureCount).ToArray();
            double[] maxs = Enumerable.Repeat(double.MinValue, featureCount).ToArray();

            for (int i = 0; i < sampleCount; i++)
            {
                for (int j = 0; j < featureCount; j++)
                {
                    double value = rawFeatures[i, j] ?? featureMedians[j];
                    cleanFeatures[i, j] = value;
                    if (value < mins[j]) mins[j] = value;
                    if (value > maxs[j]) maxs[j] = value;
                }
            }

            // 3. Min-Max Normalization to [0, 1]
            List<DataPoint> data = new List<DataPoint>();
            for (int i = 0; i < sampleCount; i++)
            {
                double[] normFeatures = new double[featureCount];
                for (int j = 0; j < featureCount; j++)
                {
                    normFeatures[j] = (cleanFeatures[i, j] - mins[j]) / (maxs[j] - mins[j]);
                }
                data.Add(new DataPoint(normFeatures, targets[i]));
            }

            // 4. Shuffle and Split
            Random rng = new Random(seed);
            var shuffledData = data.OrderBy(_ => rng.Next()).ToArray();

            int trainCount = (int)(sampleCount * trainRatio);
            var train = shuffledData.Take(trainCount).ToArray();
            var test = shuffledData.Skip(trainCount).ToArray();

            return (train, test);
        }
    }
}
