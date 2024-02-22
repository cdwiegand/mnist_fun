using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mnistfun
{
    public class Trainer : BaseModel
    {
        public Trainer(Dictionary<int, char> vectorMapping) : base(vectorMapping) { }

        public LayerChain BuildTrainingLayers(List<Tuple<char, double[]>> sourceData)
        {
            int maxCountPixels = sourceData.Max(p => p.Item2.Length); // should be 784

            // now, run through ML..
            LayerChain layers = new LayerChain();
            layers.Add(new Layer(maxCountPixels));
            //layers.Add(new Layer((int)(maxCountPixels / Math.Log(maxCountPixels)))); // roughly 80 or so if MNIST
            layers.Add(new Layer((int)Math.Sqrt(maxCountPixels))); // 28 if MNIST
            layers.Add(new Layer(sourceData.Select(p => p.Item1).Distinct().Count())); // 10 if mnist if MNIST

            return layers;
        }

        public SourceData LoadTraining(Args theArgs)
        {
            SourceData sourceData;

            string FILL_LETTERS = "1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt";
            for (int i = 0; i < FILL_LETTERS.Length; i++)
                VECTOR_MAPPING[i] = FILL_LETTERS[i];

            if (string.IsNullOrEmpty(theArgs.TrainingPath) || theArgs.TrainingPath == "." || theArgs.TrainingPath == "-")
            {
                // assume mnist
                string dirPath = FindPath(p => System.IO.File.Exists(System.IO.Path.Combine(p, "training", "0", "1.png")));

                if (dirPath == null)
                    throw new Exception("Please download the MNIST images, as pngs, such that there is a path training\\0\\1.png (or training/0/1.png)");

                Console.WriteLine($"Using MNIST images from {dirPath}... please wait!");
                sourceData = LoadPngImages(System.IO.Path.Join(dirPath, "training"));
            }
            else if (System.IO.File.Exists(theArgs.TrainingPath))
                sourceData = LoadCsvFiles(theArgs.TrainingPath);
            else
                throw new Exception("Don't know what to do!");

            return sourceData;
        }

        public void RunTraining(LayerChain layers, SourceData sourceData)
        {
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
                {
                    Console.WriteLine(epochVal);
                }

                epoch++;
                if (epoch >= 3) Console.WriteLine("Reply 'N' to quit.");
            } while (epoch < 3 || !(Console.ReadLine() ?? "").Trim().ToLower().StartsWith("n"));
        }

        private SourceData LoadCsvFiles(string loadPath)
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

            return new SourceData(tempData.Select(p => new Tuple<char, double[]>(DevectorizeKey(p.Item1), p.Item2)));
        }

        private static string FindPath(Func<string, bool> Test)
        {
            string dirPath = System.Environment.CurrentDirectory;
            while (dirPath != null && !Test(dirPath) && System.IO.Path.GetPathRoot(dirPath) != dirPath)
                dirPath = System.IO.Directory.GetParent(dirPath)?.FullName;
            return dirPath;
        }

        private static SourceData LoadPngImages(string dirPath)
        {
            var mnist_data = new SourceData();
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
    }
}
