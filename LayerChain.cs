using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static mnistfun.Program;

namespace mnistfun
{
    public class LayerChain
    {
        public LinkedList<InterLayerMatrix> Matrices = new LinkedList<InterLayerMatrix>();
        public LinkedList<Layer> Layers = new LinkedList<Layer>();
        public Layer Output => Layers.Last.Value;

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

        public static LayerChain LoadModel(RuntimeConfig config)
        {
            string json = System.IO.File.ReadAllText(config.ModelFile);
            LayerChain ret = System.Text.Json.JsonSerializer.Deserialize<LayerChain>(json);
            return ret;
        }

        public void SaveModel(RuntimeConfig config)
        {
            string json = System.Text.Json.JsonSerializer.Serialize(this);
            System.IO.File.WriteAllText(config.ModelFile, json);
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

        public double[] GenerateOutputDelta(int properMatchingOutputNeuron)
        {
            var output = Layers.Last.Value;
            double[] deltaOutput = new double[output.Length];
            for (int i = 0; i < output.neurons.Length; i++)
                deltaOutput[i] = (output.neurons[i] - (i == properMatchingOutputNeuron ? 1 : 0));
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
