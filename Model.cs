using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json.Nodes;
using System.Text.Json;
using System.Threading.Tasks;
using static mnistfun.Program;

namespace mnistfun
{
    [Serializable]
    public class Model
    {
        public LinkedList<InterLayerMatrix> Matrices { get; set; } = new LinkedList<InterLayerMatrix>();
        public LinkedList<Layer> Layers { get; set; } = new LinkedList<Layer>();
        public Layer Output => Layers.Last.Value;
        public List<TrainingLoopResult> TrainingLoops { get; set; } = new List<TrainingLoopResult>();

        public Model Add(Layer layer)
        {
            Layers.AddLast(layer);
            if (Layers.Count >= 2)
                Matrices.AddLast(new InterLayerMatrix(Layers.Last.Previous.Value, Layers.Last.Value));

            return this;
        }
        public Layer GetLayer(Guid id) => Layers.First(p => p.Id == id);

        public JsonNode ToJson()
        {
            JsonObject root = new JsonObject();
            root.Add("TrainingLoops", new JsonArray(TrainingLoops.Select(p => p.ToJson()).ToArray()));
            root.Add("Layers", new JsonArray(Layers.Select(p => p.ToJson()).ToArray()));
            root.Add("Matrices", new JsonArray(Matrices.Select(p => p.ToJson()).ToArray()));
            return root;
        }
        public static Model FromJson(JsonNode json)
        {
            var ret = new Model();
            foreach (JsonNode jn in json["Layers"]?.AsArray())
                ret.Layers.AddLast(Layer.FromJson(jn));
            foreach (JsonNode jn in json["Matrices"]?.AsArray())
                ret.Matrices.AddLast(InterLayerMatrix.FromJson(jn, ret));
            foreach (JsonNode jn in json["TrainingLoops"]?.AsArray())
                ret.TrainingLoops.Add(TrainingLoopResult.FromJson(jn));
            return ret;
        }

        public static Model LoadModel(RuntimeConfig config)
        {
            string json = System.IO.File.ReadAllText(config.ModelFile);
            var root = JsonObject.Parse(json);
            Model ret = FromJson(root);
            return ret;
        }

        public void SaveModel(RuntimeConfig config) => System.IO.File.WriteAllText(config.ModelFile, ToJson().ToString());

        public void SetInputNeurons(double[] input) => Layers.First.Value.SetNeurons(input);

        public void ApplyMatricesForward()
        {
            foreach (var m in Matrices)
                m.ApplyMatricesForward();
        }

        public double[] GenerateOutputDelta(int properMatchingOutputNeuron)
        {
            var output = Layers.Last.Value;
            double[] deltaOutput = new double[output.Length];
            for (int i = 0; i < output.Length; i++)
                deltaOutput[i] = (output.Neurons[i] - (i == properMatchingOutputNeuron ? 1 : 0));
            return deltaOutput;
        }

        public void ApplyOutputDelta(double[] deltaOutput)
        {
            var hidden1 = Layers.Last.Previous.Value;
            var hidden1ToOutput = Matrices.Last.Value;
            // backpropagate from output layer to hidden layer 1 learning adjustments
            var outputToHiddenLayer1Correction = hidden1.CalculateBackpropagationMatrix(deltaOutput, -0.01);
            hidden1ToOutput.ApplyBackPropagationMatrix(outputToHiddenLayer1Correction);
        }

        public void BackpropagateDelta(double[] deltaOutput)
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
}
