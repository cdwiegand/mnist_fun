using CsvHelper;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System.Net.WebSockets;
using System.Reflection;
using System.Reflection.Metadata;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json.Serialization;
using System.Text.Json;

namespace mnistfun
{
    internal class Program
    {
        static internal readonly Random rand = new Random();
        internal static readonly JsonSerializerOptions DefaultJsonSerializeOptions = new JsonSerializerOptions
        {
            Converters = {
                new JsonStringEnumConverter()
            },
            DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull | System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingDefault
        };

        static void Main(string[] args)
        {
            RuntimeConfig config = new RuntimeConfig(args);
            Console.WriteLine("Using config: " + config);

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