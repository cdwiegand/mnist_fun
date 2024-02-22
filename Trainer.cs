using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mnistfun
{
    public class Trainer
    {
        public Trainer(RuntimeConfig config) { this.config = config; }
        private readonly RuntimeConfig config;

        public LayerChain BuildTrainingLayers(SourceData sourceData)
        {
            int maxCountPixels = sourceData.Max(p => p.Pixels.Length); // should be 784

            // now, run through ML..
            LayerChain layers = new LayerChain();
            layers.Add(new Layer(maxCountPixels));

            if (config.RequestedHiddenLayers == null || config.RequestedHiddenLayers.Length == 0)
            {
                layers.Add(new Layer((int)(maxCountPixels / Math.Log(maxCountPixels)))); // roughly 80 or so if MNIST
                layers.Add(new Layer((int)Math.Sqrt(maxCountPixels))); // 28 if MNIST
            }
            else
            {
                foreach (int i in config.RequestedHiddenLayers)
                    layers.Add(new Layer(i)); // config already confirmed they are ints
            }

            layers.Add(new Layer(sourceData.Select(p => p.Character).Distinct().Count())); // 10 if mnist if MNIST

            return layers;
        }

        public void RunTraining(LayerChain layers, SourceData sourceData)
        {
            var epochsTimesRightWrongs = new List<EpochResult>();
            int epoch = 0;

            do
            {
                var res = new EpochResult() { EpochGeneration = epoch };

                foreach (var item in sourceData.OrderBy(p => Guid.NewGuid()))
                {
                    int properMatchingOutputNeuron = config.VectorizeKey(item.Character);
                    layers.SetInputNeurons(item.Pixels);
                    layers.ApplyMatricesForward();

                    // ok, figure out error cost
                    var deltaOutput = layers.GenerateOutputDelta(properMatchingOutputNeuron);
                    layers.ApplyOutputDelta(deltaOutput);
                    layers.BackpropagateDelta(deltaOutput);

                    // ok, did the output match?
                    int matchedOutputNeuron = layers.Output.FindHighestValueOutputNeuron();
                    char matchedOutputChar = config.DevectorizeKey(matchedOutputNeuron);
                    if (matchedOutputChar == item.Character)
                        res.CountRight(item.Character);
                    else
                        res.CountWrong(item.Character);
                }
                epochsTimesRightWrongs.Add(res);
                Console.WriteLine(res);
                epoch++;
            } while (epoch < config.Loops);
        }
    }
}
