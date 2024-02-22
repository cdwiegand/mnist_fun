using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json.Nodes;
using System.Text.Json;
using System.Threading.Tasks;

namespace mnistfun
{
    [Serializable]
    public class Layer
    {
        [Obsolete("Do not use - this is for serialization only.")]
        public Layer() { }

        public Layer(int countNeurons)
        {
            Neurons = new double[countNeurons];
            MathUtil.FillWithRandom(Program.rand, Neurons);
        }
        public void SetNeurons(double[] values)
        {
            Neurons = values;
        }

        public Guid Id { get; set; } = Guid.NewGuid(); // default
        public double[] Neurons { get; set; }

        public int Length => Neurons.Length;
        public int Count => Neurons.Length;

        public JsonNode ToJson()
        {
            JsonObject root = new JsonObject();
            root.Add("Id", Id);
            root.Add("Neurons", JsonSerializer.SerializeToNode(Neurons));
            return root;
        }
        public static Layer FromJson(JsonNode json)
        {
            var ret = new Layer();
            ret.Id = json["Id"].Deserialize<Guid>();
            ret.Neurons = json["Neurons"].Deserialize<double[]>();
            return ret;
        }

        public void Reset()
        {
            Neurons = new double[Length];
        }

        public double[,] CalculateBackpropagationMatrix(double[] errorCost, double learnRate)
        {
            if (learnRate > 0) learnRate = 0 - learnRate; // needs to be negative
            double[,] ret = new double[Length, errorCost.Length];
            for (int neuron = 0; neuron < Length; neuron++)
                for (int errorIdx = 0; errorIdx < errorCost.Length; errorIdx++)
                    ret[neuron, errorIdx] = Neurons[neuron] * errorCost[errorIdx] * learnRate; // learn rate should be like -0.01
            return ret;
        }

        public int FindHighestValueOutputNeuron()
        {
            double highestValue = double.MinValue;
            int bestIdx = -1;
            for (int idx = 0; idx < Neurons.Length; idx++)
                if (Neurons[idx] > highestValue)
                {
                    highestValue = Neurons[idx];
                    bestIdx = idx;
                }

            return bestIdx;
        }
    }
}
