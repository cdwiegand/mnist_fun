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
        private Layer() { }

        public Layer(int countNeurons)
        {
            Neurons = MathUtil.InitializeSingleMatrix(countNeurons);
            MathUtil.FillWithRandom(Program.rand, Neurons);
        }
        public void SetNeurons(double[] values)
        {
            for (int i = 0; i < values.Length; i++) Neurons[i] = values[i];
        }

        public Guid Id { get; set; } = Guid.NewGuid(); // default
        public Dictionary<int, double> Neurons { get; set; } = new Dictionary<int, double>();

        public int Length => Neurons.Count;

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
            ret.Neurons = json["Neurons"].Deserialize<Dictionary<int, double>>();
            return ret;
        }

        public void Reset() => Neurons = Neurons.Select(p => new KeyValuePair<int, double>(p.Key, 0)).ToDictionary(p => p.Key, p => p.Value);

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
            double highestValue = Neurons.Values.Max();
            int idx = 0;
            while (Neurons[idx] < highestValue) idx++; // have to iterate
            return idx;
        }
    }
}
