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

        public void Run(SourceData sourceData, Model model)
        {
            foreach (var item in sourceData.OrderBy(p => Guid.NewGuid()))
            {
                int properMatchingOutputNeuron = config.VectorizeKey(item.Character);
                model.SetInputNeurons(item.Pixels);
                model.ApplyMatricesForward();

                // ok, did the output match?
                int matchedOutputNeuron = model.Output.FindHighestValueOutputNeuron();
                char matchedOutputChar = config.DevectorizeKey(matchedOutputNeuron);

                Console.WriteLine($"{matchedOutputChar} from {item.LoadPath}");
            }
        }

    }
}
