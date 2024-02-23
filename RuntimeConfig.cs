using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace mnistfun
{
    public class RuntimeConfig
    {
        private static Dictionary<int, char> VECTOR_MAPPING = new Dictionary<int, char>();

        public RuntimeConfig(string[] args)
        {
            TrainingPath = GetIfArg(args, "train");
            RunFile = GetIfArg(args, "run");
            ModelFile = GetIfArg(args, "model");
            RuntimeMode mode = GetIfArg(args, "mode", p => Enum.TryParse(p, true, out RuntimeMode result) ? result : RuntimeMode.Detect);
            Loops = GetIfArg(args, "loops", p => int.TryParse(p, out int result) ? result : 3);
            string? hiddenlayers = GetIfArg(args, "hiddenlayers");

            if (mode == RuntimeMode.Detect)
            {
                // can we determine?
                if (!string.IsNullOrEmpty(TrainingPath)) Mode = RuntimeMode.Training;
                else if (!string.IsNullOrEmpty(RunFile)) Mode = RuntimeMode.Running;
                else throw new Exception("No valid --mode: specified, and can't infer usage!");
            }

            if (!string.IsNullOrEmpty(hiddenlayers))
            {
                // means we want a specific size setup
                var tmp = hiddenlayers.Split(',', ':'); // 100:64 as an example or 40,10
                if (!tmp.All(p => int.TryParse(p, out _)))
                    throw new Exception("All hidden layer values --hiddenlayers:x,y,z must be ints!");
                RequestedHiddenLayers = tmp.Select(p => int.Parse(p)).ToArray();
            }

            // valid?
            if (string.IsNullOrEmpty(TrainingPath) && string.IsNullOrEmpty(RunFile))
                throw new Exception("Must specify at least a training path or a run file!");
            if (!string.IsNullOrEmpty(RunFile) && (string.IsNullOrEmpty(ModelFile) || !System.IO.File.Exists(ModelFile)))
                throw new Exception("If running, must specify a VALID model file!");

            string FILL_LETTERS = "1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt";
            for (int i = 0; i < FILL_LETTERS.Length; i++)
                VECTOR_MAPPING[i] = FILL_LETTERS[i];
        }

        private static string? GetIfArg(string[] args, string name, string? defaultValue = null)
        {
            string? arg = args.FirstOrDefault(p => p.StartsWith($"--{name}:") || p.StartsWith($"--{name}="));
            if (string.IsNullOrEmpty(arg) && args.Any(p => p == "--" + name))
            {
                // try the other form: --name valueHere
                for (int i = 0; i < args.Length - 1; i++)
                    if (args[i] == "--" + name) return args[i + 1];
            }
            if (string.IsNullOrEmpty(arg)) return defaultValue; // not present

            int idxColon = arg.IndexOf(':');
            int idxEqual = arg.IndexOf('=');
            if (idxColon > -1 && idxEqual > -1)
                return arg.Substring(Math.Min(idxColon, idxEqual) + 1);
            else if (idxColon > -1)
                return arg.Substring(idxColon + 1);
            else if (idxEqual > -1)
                return arg.Substring(idxEqual + 1);
            else
                return defaultValue;
        }
        private static T? GetIfArg<T>(string[] args, string name, Func<string, T> conv)
        {
            string found = GetIfArg(args, name);
            if (!string.IsNullOrEmpty(found)) return conv(found);
            return default(T);
        }

        public string? TrainingPath { get; private set; }
        public string? RunFile { get; private set; }
        public string? ModelFile { get; private set; }
        public RuntimeMode Mode { get; private set; }
        public int Loops { get; private set; }
        public int[] RequestedHiddenLayers { get; private set; }

        public override string ToString() => System.Text.Json.JsonSerializer.Serialize(this, Program.DefaultJsonSerializeOptions);

        public enum RuntimeMode
        {
            Detect,
            Training,
            Running,
            TrainMore,
        }

        // given an index (vector), what is the character?
        public char DevectorizeKey(int idx) => VECTOR_MAPPING.FirstOrDefault(p => p.Key == idx).Value;

        // basicly, turn a character into its index (or vector) for comparison
        public int VectorizeKey(char key) => VECTOR_MAPPING.FirstOrDefault(p => p.Value == key).Key;
    }
}
