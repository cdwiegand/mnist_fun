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

        static void Main(string[] args)
        {
            RuntimeConfig config = new RuntimeConfig(args);

            DateTime utcStart = DateTime.UtcNow;
            switch (config.Mode)
            {
                case RuntimeConfig.RuntimeMode.Running:
                    {
                        Runner runner = new Runner(config);
                        SourceData source = SourceData.LoadRunningSource(config.RunFile, config);
                        LayerChain chains = LayerChain.LoadModel(config);

                        Console.WriteLine($"Loaded {source.Count} model data, now analysing...");
                        runner.Run(source, chains);
                    }
                    break;
                case RuntimeConfig.RuntimeMode.Training:
                    {
                        Trainer trainer = new Trainer(config);
                        SourceData source = SourceData.LoadTrainingSource(config.TrainingPath, config);
                        LayerChain chains = trainer.BuildTrainingLayers(source);

                        Console.WriteLine($"Loaded {source.Count} model data, now analysing...");
                        trainer.RunTraining(chains, source);

                        if (!string.IsNullOrEmpty(config.ModelFile))
                            chains.SaveModel(config);
                    }
                    break;
                default: throw new NotImplementedException("Unknown mode value: " + config.Mode);
            }
            TimeSpan ts = DateTime.UtcNow.Subtract(utcStart);
            Console.WriteLine($"Took {ts} time to run.");
        }
    }
}