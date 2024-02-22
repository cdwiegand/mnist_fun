using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mnistfun
{
    public class Runner
    {
        public Runner(RuntimeConfig config) { this.config = config; }
        private readonly RuntimeConfig config;

        public void Run(SourceData sourceData, LayerChain layers)
        {
            foreach (var item in sourceData.OrderBy(p => Guid.NewGuid()))
            {
                int properMatchingOutputNeuron = config.VectorizeKey(item.Character);
                layers.SetInputNeurons(item.Pixels);
                layers.ApplyMatricesForward();

                // ok, did the output match?
                int matchedOutputNeuron = layers.Output.FindHighestValueOutputNeuron();
                char matchedOutputChar = config.DevectorizeKey(matchedOutputNeuron);

                Console.WriteLine($"{item.Character} from {item.LoadPath}");
            }
        }

    }
}
