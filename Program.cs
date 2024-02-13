using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.Processing;
using System.Reflection;
using System.Reflection.Metadata;

namespace mnistfun
{
    internal class Program
    {
        static void Main(string[] args)
        {
            string dirPath = System.Environment.CurrentDirectory;
            while (dirPath != null && !System.IO.File.Exists(System.IO.Path.Combine(dirPath, "training", "0", "1.png")) && System.IO.Path.GetPathRoot(dirPath) != dirPath)
                dirPath = System.IO.Directory.GetParent(dirPath)?.FullName;

            if (dirPath == null)
                throw new Exception("Please download the MNIST images, as pngs, such that there is a path training\\0\\1.png");

            Console.WriteLine($"Using MNIST images from {dirPath}... please wait!");
            List<Tuple<int, double[]>> mnist_data = new List<Tuple<int, double[]>>();
            for (int i = 0; i < 10; i++)
            {
                var files = System.IO.Directory.GetFiles($"c:\\users\\chris\\downloads\\pyy\\out\\training\\{i}");
                foreach (string file in files)
                {
                    var image = SixLabors.ImageSharp.Image.Load<SixLabors.ImageSharp.PixelFormats.Argb32>(file);
                    double[] pixels = new double[image.Width * image.Height];
                    int idx = 0;
                    for (int y = 0; y < image.Height; y++)
                        for (int x = 0; x < image.Width; x++)
                        {
                            double pixel = ((double)image[x, y].R) / 256;
                            pixels[idx++] = pixel; // assign and more to next pixel
                        }
                    mnist_data.Add(new Tuple<int, double[]>(i, pixels));
                }
            }
            int maxCountPixels = mnist_data.Max(p => p.Item2.Length); // should be 784

            // now, run through ML..
            // need 784 input parameters, 10 output parameters, so ... ?? ... 30 hidden (1st layer) parameters?
            int inputParamCount = maxCountPixels;
            int hiddenLayer1Count = 30;
            int outputParamCount = 10;
            double[,] inputToHiddenLayer1Matrix = new double[inputParamCount, hiddenLayer1Count];
            double[,] hiddenLayer1ToOutputMatrix = new double[hiddenLayer1Count, outputParamCount];

            var rand = new Random();
            FillWithRandom(rand, inputToHiddenLayer1Matrix);
            FillWithRandom(rand, hiddenLayer1ToOutputMatrix);
            /*double[] biasInputToHidden = new double[hiddenLayer1Count];
            double[] biasHiddenToOutput = new double[outputParamCount];
            FillWithValue(0, biasInputToHidden);
            FillWithValue(0, biasHiddenToOutput);*/

            var epochsTimesRightWrongs = new List<EpochResult>();
            int epoch = 0;

            Console.WriteLine($"Loaded {mnist_data.Count} images, now analysing...");
            do
            {
                var res = new EpochResult() { EpochGeneration = epoch };

                foreach (var item in mnist_data.OrderBy(p => Guid.NewGuid()))
                {
                    var hiddenLayer1Values = ApplyMatricesForward(inputToHiddenLayer1Matrix, item.Item2);
                    var outputLayerValues = ApplyMatricesForward(hiddenLayer1ToOutputMatrix, hiddenLayer1Values);

                    // ok, figure out error cost
                    double[] deltaOutput = new double[outputParamCount];
                    for (int i = 0; i < outputParamCount; i++)
                        deltaOutput[i] = (outputLayerValues[i] - (i == item.Item1 ? 1 : 0));
                    var errCost = 1 / outputLayerValues.Length * deltaOutput.Sum(p => Math.Pow(p, 2));

                    // backpropagate from output layer to hidden layer 1 learning adjustments
                    var outputToHiddenLayer1Correction = CalculateBackpropagationMatrix(deltaOutput, hiddenLayer1Values, -0.01);
                    hiddenLayer1ToOutputMatrix = ApplyBackPropagationMatrix(outputToHiddenLayer1Correction, hiddenLayer1ToOutputMatrix);

                    // harder math, can't cheat
                    double[] deltaHiddenLayer1a = new double[hiddenLayer1Count];
                    for (int i = 0; i < hiddenLayer1Count; i++)
                        deltaHiddenLayer1a[i] = (hiddenLayer1Values[i] * (1 - hiddenLayer1Values[i]));
                    double[] deltaHiddenLayer1b = MatrixMultiply(hiddenLayer1ToOutputMatrix, deltaOutput);
                    double[] deltaHiddenLayer1c = new double[hiddenLayer1Count];
                    for (int i = 0; i < deltaHiddenLayer1c.Length; i++)
                        deltaHiddenLayer1c[i] = deltaHiddenLayer1b[i] * deltaHiddenLayer1a[i];

                    // ok, now adjust input weights
                    var hiddenLayer1ToInputCorrection = CalculateBackpropagationMatrix(deltaHiddenLayer1c, item.Item2, -0.01);
                    inputToHiddenLayer1Matrix = ApplyBackPropagationMatrix(hiddenLayer1ToInputCorrection, inputToHiddenLayer1Matrix);

                    // ok, did the output match?
                    int selected = FoundHighestValue(outputLayerValues);
                    if (selected == item.Item1)
                    {
                        //Console.WriteLine($"YAY I found {selected}");
                        Console.Write($" {selected}{item.Item1}!");
                        res.CountRight(item.Item1);
                    }
                    else
                    {
                        //Console.WriteLine($"FAILED: I thought it was {selected} but it was {item.Item1}");
                        Console.Write($" {selected}{item.Item1} .");
                        res.CountWrong(item.Item1);
                    }
                }
                epochsTimesRightWrongs.Add(res);

                Console.WriteLine();
                foreach (var epochVal in epochsTimesRightWrongs)
                    Console.WriteLine(epochVal);

                epoch++;
                if (epoch >= 3) Console.WriteLine("Reply 'N' to quit.");
            } while (epoch < 3 || !(Console.ReadLine() ?? "").Trim().ToLower().StartsWith("n"));
        }

        public class EpochResult
        {
            public EpochResult()
            {
                for (int i = 0; i < Numbers.Length; i++)
                    Numbers[i] = new CorrectNumberGuessed();
            }

            public int EpochGeneration;
            public CorrectNumberGuessed[] Numbers = new CorrectNumberGuessed[10];
            public int CountedCorrect => Numbers.Sum(p => p.CountedRight);
            public int CountedWrong => Numbers.Sum(p => p.CountedWrong);
            public void CountRight(int number) => Numbers[number].CountRight();
            public void CountWrong(int number) => Numbers[number].CountWrong();

            public override string ToString()
            {
                string ret = $"Epoch: {EpochGeneration}:";
                for (int i = 0; i < 10; i++)
                    ret += $" {i}: {Numbers[i].PercentStr}";
                return ret;
            }
        }
        public class CorrectNumberGuessed
        {
            public int CountedRight = 0;
            public int CountedWrong = 0;
            public int Total => CountedRight + CountedWrong;
            public string PercentStr => Total > 0 ? ((decimal)CountedRight / Total).ToString("P2") : "N/A";
            public void CountRight() => CountedRight += 1;
            public void CountWrong() => CountedWrong += 1;
        }

        static int FoundHighestValue(double[] values)
        {
            double highestValue = double.MinValue;
            int bestIdx = -1;
            for (int idx = 0; idx < values.Length; idx++)
            {
                if (values[idx] > highestValue)
                {
                    highestValue = values[idx];
                    bestIdx = idx;
                }
            }
            return bestIdx;
        }

        static double[] MatrixMultiply(double[,] data, double[] columnMultipliers)
        {
            if (columnMultipliers.Length != data.GetLength(1)) throw new Exception("Bad matrix!");

            double[] ret = new double[data.GetLength(0)];
            FillWithValue(0, ret); // initialize to zero

            for (int i = 0; i < data.GetLength(0); i++)
                for (int j = 0; j < data.GetLength(1); j++)
                    ret[j] += data[i, j] * columnMultipliers[j];
            return ret;
        }

        static double[,] ApplyBackPropagationMatrix(double[,] correctionMatrix, double[,] weightMatrix)
        {
            // should be the same shape (dimensions) so easy!
            for (int i = 0; i < correctionMatrix.GetLength(0); i++)
                for (int j = 0; j < correctionMatrix.GetLength(1); j++)
                    weightMatrix[i, j] += correctionMatrix[i, j];
            return weightMatrix;
        }

        static double[,] CalculateBackpropagationMatrix(double[] errorCost, double[] neuronValues, double learnRate)
        {
            if (learnRate > 0) learnRate = 0 - learnRate; // needs to be negative
            double[,] ret = new double[neuronValues.Length, errorCost.Length];
            for (int neuron = 0; neuron < neuronValues.Length; neuron++)
                for (int errorIdx = 0; errorIdx < errorCost.Length; errorIdx++)
                    ret[neuron, errorIdx] = neuronValues[neuron] * errorCost[errorIdx] * learnRate; // learn rate should be like -0.01
            return ret;
        }

        static double[] ApplyMatricesForward(double[,] weightMatrix, double[] values)
        {
            // weight matrix 1st dimension is left/input layer (rows), 2nd dimension is right/output layer (column)
            // (note: opposite of youtube video I'm watching!)
            int columns = weightMatrix.GetLength(1);
            double[] resultValues = new double[columns];
            for (int resultIdx = 0; resultIdx < columns; resultIdx++)
            {
                double h_pre = 0;
                for (int pix = 0; pix < weightMatrix.GetLength(0); pix++)
                    // multiply them by the input -> hidden layer 1, then apply sigmoid function to that                            
                    h_pre += values[pix] * weightMatrix[pix, resultIdx];
                resultValues[resultIdx] = Sigmoid(h_pre);
            }
            return resultValues;
        }

        static double Sigmoid(double h_pre) => (double)(1 / (1 + Math.Pow(Math.E, (double)(-h_pre))));

        static void FillWithRandom(Random rand, double[] matrix)
        {
            for (int i2hl1 = 0; i2hl1 < matrix.GetLength(0); i2hl1++)
                matrix[i2hl1] = rand.NextDouble() - 0.5;
        }
        static void FillWithValue(double value, double[] matrix)
        {
            for (int i2hl1 = 0; i2hl1 < matrix.GetLength(0); i2hl1++)
                matrix[i2hl1] = value;
        }
        static void FillWithRandom(Random rand, double[,] matrix)
        {
            for (int i = 0; i < matrix.GetLength(0); i++)
                for (int j = 0; j < matrix.GetLength(1); j++)
                    matrix[i, j] = rand.NextDouble() - 0.5;
        }
        static void FillWithValue(double value, double[,] matrix)
        {
            for (int i = 0; i < matrix.GetLength(0); i++)
                for (int j = 0; j < matrix.GetLength(1); j++)
                    matrix[i, j] = value;
        }
        static void FillWithValue(int value, int[] matrix)
        {
            for (int i2hl1 = 0; i2hl1 < matrix.GetLength(0); i2hl1++)
                matrix[i2hl1] = value;
        }
    }
}