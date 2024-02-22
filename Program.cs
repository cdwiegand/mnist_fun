using CsvHelper;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System.Net.WebSockets;
using System.Reflection;
using System.Reflection.Metadata;
using System.Runtime.CompilerServices;
using System.Text;

namespace mnistfun
{
    internal class Program
    {
        static internal readonly Random rand = new Random();
        private static Dictionary<int, char> VECTOR_MAPPING = new Dictionary<int, char>();

        static void Main(string[] args)
        {
            Args theArgs = new Args(args);

            switch (theArgs.Mode)
            {
                case Args.RuntimeMode.Running:
                    {
                        LayerChain chains = LayerChain.LoadModel(theArgs);
                        // FIXME run
                    }
                    break;
                case Args.RuntimeMode.Training:
                    {
                        Trainer trainer = new Trainer(VECTOR_MAPPING);
                        SourceData trainingData = trainer.LoadTraining(theArgs);
                        LayerChain chains = trainer.BuildTrainingLayers(trainingData);
                        trainer.RunTraining(chains, trainingData);
                        if (!string.IsNullOrEmpty(theArgs.ModelFile))
                            chains.SaveModel(theArgs);
                    }
                    break;
                default: throw new NotImplementedException("Unknown mode value: " + theArgs.Mode);
            }
        }
    }
}