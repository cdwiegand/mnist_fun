using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mnistfun
{
    public class Layer
    {
        public Layer(int countNeurons)
        {
            neurons = new double[countNeurons];
            MathUtil.FillWithRandom(Program.rand, neurons);
        }
        public void SetNeurons(double[] values)
        {
            neurons = values;
        }

        public double[] neurons;
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
}
