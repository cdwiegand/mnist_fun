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

        public Model BuildTrainingLayers(SourceData sourceData)
        {
            int maxCountPixels = sourceData.Max(p => p.Pixels.Length); // should be 784

            // now, run through ML..
            Model model = new Model();
            model.Add(new Layer(maxCountPixels));

            if (config.RequestedHiddenLayers == null || config.RequestedHiddenLayers.Length == 0)
            {
                model.Add(new Layer((int)(maxCountPixels / Math.Log(maxCountPixels)))); // roughly 80 or so if MNIST
                model.Add(new Layer((int)Math.Sqrt(maxCountPixels))); // 28 if MNIST
            }
            else
            {
                foreach (int i in config.RequestedHiddenLayers)
                    model.Add(new Layer(i)); // config already confirmed they are ints
            }

            model.Add(new Layer(sourceData.Select(p => p.Character).Distinct().Count())); // 10 if mnist if MNIST

            return model;
        }

        public void RunTraining(Model model, SourceData sourceData)
        {
            int epoch = model.TrainingLoops.Count;

            while (epoch < config.Loops && (!config.StopIfEveryoneMinQuality.HasValue || model.TrainingLoops.Count == 0 || model.TrainingLoops.Any(p => p.WorstAccuracy < config.StopIfEveryoneMinQuality.Value)))
            {
                var res = new TrainingLoopResult() { LoopGeneration = epoch, StartTime = DateTime.UtcNow };

                foreach (var item in sourceData.OrderBy(p => Guid.NewGuid()))
                {
                    int properMatchingOutputNeuron = config.VectorizeKey(item.Character);
                    model.SetInputNeurons(item.Pixels);
                    model.ApplyMatricesForward();

                    // ok, figure out error cost
                    var deltaOutput = model.GenerateOutputDelta(properMatchingOutputNeuron);
                    model.ApplyOutputDelta(deltaOutput);
                    model.BackpropagateDelta(deltaOutput);

                    // ok, did the output match?
                    int matchedOutputNeuron = model.Output.FindHighestValueOutputNeuron();
                    char matchedOutputChar = config.DevectorizeKey(matchedOutputNeuron);
                    if (matchedOutputChar == item.Character)
                        res.CountRight(item.Character);
                    else
                        res.CountWrong(item.Character);
                }

                res.EndTime = DateTime.UtcNow;
                model.TrainingLoops.Add(res);
                Console.WriteLine(res);
                epoch++;
            }
        }
    }
}
