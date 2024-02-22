using CsvHelper;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System.Net.WebSockets;
using System.Reflection;
using System.Reflection.Metadata;
using System.Runtime.CompilerServices;
using System.Text;

namespace mnistfun
{
    internal class Program
    {
        static internal readonly Random rand = new Random();
        private static Dictionary<int, char> VECTOR_MAPPING = new Dictionary<int, char>();

        static void Main(string[] args)
        {
            List<Tuple<char, double[]>> sourceData;

            string FILL_LETTERS = "1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt";
            for (int i = 0; i < FILL_LETTERS.Length; i++)
                VECTOR_MAPPING[i] = FILL_LETTERS[i];

            if (args.Length == 0)
            {
                // assume mnist
                string dirPath = FindPath(p => System.IO.File.Exists(System.IO.Path.Combine(p, "training", "0", "1.png")));

                if (dirPath == null)
                    throw new Exception("Please download the MNIST images, as pngs, such that there is a path training\\0\\1.png (or training/0/1.png)");

                Console.WriteLine($"Using MNIST images from {dirPath}... please wait!");
                sourceData = LoadPngImages(System.IO.Path.Join(dirPath, "training"));
            }
            else if (args.Length == 1)
            {
                string dirPath = System.IO.File.Exists(args[0]) ? args[0] : FindPath(p => System.IO.File.Exists(System.IO.Path.Combine(p, args[0])));
                if (dirPath == null)
                    throw new Exception("File not found.");
                string loadPath = System.IO.File.Exists(args[0]) ? args[0] : System.IO.Path.Combine(dirPath, args[0]);

                sourceData = LoadCsvFiles(loadPath);
            }
            else
                throw new Exception("Don't know what to do!");

            int maxCountPixels = sourceData.Max(p => p.Item2.Length); // should be 784

            // now, run through ML..
            LayerChain layers = new LayerChain();
            layers.Add(new Layer(maxCountPixels));
            //layers.Add(new Layer((int)(maxCountPixels / Math.Log(maxCountPixels)))); // roughly 80 or so if MNIST
            layers.Add(new Layer((int)Math.Sqrt(maxCountPixels))); // 28 if MNIST
            layers.Add(new Layer(sourceData.Select(p => p.Item1).Distinct().Count())); // 10 if mnist if MNIST

            var epochsTimesRightWrongs = new List<EpochResult>();
            int epoch = 0;

            Console.WriteLine($"Loaded {sourceData.Count} images, now analysing...");
            do
            {
                var res = new EpochResult() { EpochGeneration = epoch };

                foreach (var item in sourceData.OrderBy(p => Guid.NewGuid()))
                {
                    int properMatchingOutputNeuron = VectorizeKey(item.Item1);
                    layers.SetInputNeurons(item.Item2);
                    layers.ApplyMatricesForward();

                    // ok, figure out error cost
                    var deltaOutput = layers.GenerateOutputDelta(properMatchingOutputNeuron);
                    layers.ApplyOutputDelta(deltaOutput);
                    layers.BackpropagateDelta(deltaOutput);

                    // ok, did the output match?
                    int matchedOutputNeuron = layers.Output.FindHighestValueOutputNeuron();
                    char matchedOutputChar = DevectorizeKey(matchedOutputNeuron);
                    if (matchedOutputChar == item.Item1)
                        res.CountRight(item.Item1);
                    else
                        res.CountWrong(item.Item1);
                }
                epochsTimesRightWrongs.Add(res);

                Console.WriteLine();
                foreach (var epochVal in epochsTimesRightWrongs)
                    Console.WriteLine(epochVal);

                epoch++;
                if (epoch >= 3) Console.WriteLine("Reply 'N' to quit.");
            } while (epoch < 3 || !(Console.ReadLine() ?? "").Trim().ToLower().StartsWith("n"));
        }

        private static List<Tuple<char, double[]>> LoadCsvFiles(string loadPath)
        {
            // path to mnist csv or emnist csv
            var tempData = new List<Tuple<int, double[]>>();
            using (var reader = new System.IO.StreamReader(loadPath))
            using (var csver = new CsvHelper.CsvReader(reader, System.Globalization.CultureInfo.InvariantCulture, false))
                while (csver.Read()) // no header!
                {
                    int fieldMap = csver.GetField<int>(0);
                    int[] pixelsInt = new int[csver.ColumnCount - 1];
                    for (int i = 1; i < csver.ColumnCount; i++)
                        pixelsInt[i - 1] = csver.GetField<int>(i);
                    double[] pixels = pixelsInt.Select(p => (double)p / 256).ToArray();
                    tempData.Add(new Tuple<int, double[]>(fieldMap, pixels));
                }

            return tempData.Select(p => new Tuple<char, double[]>(DevectorizeKey(p.Item1), p.Item2)).ToList();
        }

        // given an index (vector), what is the character?
        private static char DevectorizeKey(int idx) => VECTOR_MAPPING.FirstOrDefault(p => p.Key == idx).Value;

        // basicly, turn a character into its index (or vector) for comparison
        private static int VectorizeKey(char key) => VECTOR_MAPPING.FirstOrDefault(p => p.Value == key).Key;

        private static string FindPath(Func<string, bool> Test)
        {
            string dirPath = System.Environment.CurrentDirectory;
            while (dirPath != null && !Test(dirPath) && System.IO.Path.GetPathRoot(dirPath) != dirPath)
                dirPath = System.IO.Directory.GetParent(dirPath)?.FullName;
            return dirPath;
        }

        private static List<Tuple<char, double[]>> LoadPngImages(string dirPath)
        {
            var mnist_data = new List<Tuple<char, double[]>>();
            foreach (var subdir in System.IO.Directory.GetDirectories(dirPath))
            {
                var files = System.IO.Directory.GetFiles(subdir);
                char c = System.IO.Path.GetDirectoryName(subdir)[0];
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
                    mnist_data.Add(new Tuple<char, double[]>(c, pixels));
                }
            }
            return mnist_data;
        }

        public class InterLayerMatrix
        {
            public InterLayerMatrix(Layer fromLayer, Layer toLayer)
            {
                FromLayer = fromLayer;
                ToLayer = toLayer;
                matrix = new double[fromLayer.neurons.Length, toLayer.neurons.Length];

                MathUtil.FillWithRandom(Program.rand, matrix);
            }
            public double[,] matrix;
            public readonly Layer FromLayer;
            public readonly Layer ToLayer;

            public void ApplyMatricesForward()
            {
                // weight matrix 1st dimension is left/input layer (rows), 2nd dimension is right/output layer (column)
                // (note: opposite of youtube video I'm watching!)
                int columns = matrix.GetLength(1);
                ToLayer.Reset();
                for (int resultIdx = 0; resultIdx < columns; resultIdx++)
                {
                    double h_pre = 0;
                    for (int pix = 0; pix < matrix.GetLength(0); pix++)
                        // multiply them by the input -> hidden layer 1, then apply sigmoid function to that                            
                        h_pre += FromLayer.neurons[pix] * matrix[pix, resultIdx];
                    ToLayer.neurons[resultIdx] = MathUtil.Sigmoid(h_pre);
                }
            }

            public double[] CalculateBackpropagationMatrixDelta(double[] deltaOutput)
            {
                double[] deltaHiddenLayer1a = new double[FromLayer.Length];
                for (int i = 0; i < FromLayer.Length; i++)
                    deltaHiddenLayer1a[i] = (FromLayer.neurons[i] * (1 - FromLayer.neurons[i]));
                double[] deltaHiddenLayer1b = MathUtil.MatrixMultiply(matrix, deltaOutput);
                double[] deltaHiddenLayer1c = new double[FromLayer.Length];
                for (int i = 0; i < deltaHiddenLayer1c.Length; i++)
                    deltaHiddenLayer1c[i] = deltaHiddenLayer1b[i] * deltaHiddenLayer1a[i];
                return deltaHiddenLayer1c;
            }

            public void ApplyBackPropagationMatrix(double[,] correctionMatrix)
            {
                // should be the same shape (dimensions) so easy!
                for (int i = 0; i < correctionMatrix.GetLength(0); i++)
                    for (int j = 0; j < correctionMatrix.GetLength(1); j++)
                        matrix[i, j] += correctionMatrix[i, j];
            }
        }

        public class LayerChain
        {
            public LinkedList<InterLayerMatrix> Matrices = new LinkedList<InterLayerMatrix>();
            public LinkedList<Layer> Layers = new LinkedList<Layer>();
            public LayerChain Add(Layer layer)
            {
                Layers.AddLast(layer);
                if (Layers.Count >= 2)
                {
                    var matrix = new InterLayerMatrix(Layers.Last.Previous.Value, Layers.Last.Value);
                    Matrices.AddLast(matrix);
                }
                return this;
            }
            public void SetInputNeurons(double[] input)
            {
                Layers.First.Value.SetNeurons(input);
            }
            public void ApplyMatricesForward()
            {
                foreach (var m in Matrices)
                    m.ApplyMatricesForward();
            }
            public Layer Output => Layers.Last.Value;

            public double[] GenerateOutputDelta(int properMatchingOutputNeuron)
            {
                var output = Layers.Last.Value;
                double[] deltaOutput = new double[output.Length];
                for (int i = 0; i < output.neurons.Length; i++)
                    deltaOutput[i] = (output.neurons[i] - (i == properMatchingOutputNeuron ? 1 : 0));
                return deltaOutput;
            }

            internal void ApplyOutputDelta(double[] deltaOutput)
            {
                var hidden1 = Layers.Last.Previous.Value;
                var hidden1ToOutput = Matrices.Last.Value;
                // backpropagate from output layer to hidden layer 1 learning adjustments
                var outputToHiddenLayer1Correction = hidden1.CalculateBackpropagationMatrix(deltaOutput, -0.01);
                hidden1ToOutput.ApplyBackPropagationMatrix(outputToHiddenLayer1Correction);
            }

            internal void BackpropagateDelta(double[] deltaOutput)
            {
                for (int i = Matrices.Count - 1; i > 0; i--)
                {
                    var backSourceMatrix = Matrices.ElementAt(i);
                    var backDestMatrix = Matrices.ElementAt(i - 1);

                    // harder math, can't cheat
                    deltaOutput = backSourceMatrix.CalculateBackpropagationMatrixDelta(deltaOutput);
                    // ok, now adjust input weights
                    var corrections = backDestMatrix.FromLayer.CalculateBackpropagationMatrix(deltaOutput, -0.01);
                    backDestMatrix.ApplyBackPropagationMatrix(corrections);
                }
            }
        }

        public class Layer
        {
            public Layer(int countNeurons)
            {
                neurons = new double[countNeurons];
                MathUtil.FillWithRandom(Program.rand, neurons);
            }
            public double[] neurons;
            public void SetNeurons(double[] values)
            {
                neurons = values;
            }

            public int Length => neurons.Length;
            public int Count => neurons.Length;

            public void Reset()
            {
                neurons = new double[Length];
            }

            public double[,] CalculateBackpropagationMatrix(double[] errorCost, double learnRate)
            {
                if (learnRate > 0) learnRate = 0 - learnRate; // needs to be negative
                double[,] ret = new double[Length, errorCost.Length];
                for (int neuron = 0; neuron < Length; neuron++)
                    for (int errorIdx = 0; errorIdx < errorCost.Length; errorIdx++)
                        ret[neuron, errorIdx] = neurons[neuron] * errorCost[errorIdx] * learnRate; // learn rate should be like -0.01
                return ret;
            }
            public int FindHighestValueOutputNeuron()
            {
                double highestValue = double.MinValue;
                int bestIdx = -1;
                for (int idx = 0; idx < neurons.Length; idx++)
                    if (neurons[idx] > highestValue)
                    {
                        highestValue = neurons[idx];
                        bestIdx = idx;
                    }

                return bestIdx;
            }
        }

        public class EpochResult
        {
            public int EpochGeneration;
            public Dictionary<char, GuessResult> Characters = new Dictionary<char, GuessResult>();
            public int CountedCorrect => Characters.Select(p => p.Value.CountedRight).Sum();
            public int CountedWrong => Characters.Select(p => p.Value.CountedWrong).Sum();
            public void CountRight(char number)
            {
                if (!Characters.ContainsKey(number))
                    Characters.Add(number, new GuessResult());
                Characters[number].CountRight();
            }
            public void CountWrong(char number)
            {
                if (!Characters.ContainsKey(number))
                    Characters.Add(number, new GuessResult());
                Characters[number].CountWrong();
            }

            public override string ToString()
            {
                string ret = $"Epoch {EpochGeneration}:";
                foreach (char i in Characters.Keys)
                    ret += $" {i}: {Characters[i].PercentStr}";
                return ret;
            }
        }

        public class GuessResult
        {
            public int CountedRight = 0;
            public int CountedWrong = 0;
            public int Total => CountedRight + CountedWrong;
            public string PercentStr => Total > 0 ? ((decimal)CountedRight / Total).ToString("P2") : "N/A";
            public void CountRight() => CountedRight += 1;
            public void CountWrong() => CountedWrong += 1;
        }
    }
}