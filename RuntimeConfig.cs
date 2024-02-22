using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
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
            string? mode = GetIfArg(args, "mode");
            string? loops = GetIfArg(args, "loops");
            string? hiddenlayers = GetIfArg(args, "hiddenlayers");

            switch ((mode ?? "").Trim().ToLower())
            {
                case "train":
                case "training":
                    Mode = RuntimeMode.Training; break;
                case "run":
                case "running":
                    Mode = RuntimeMode.Running; break;
                default:
                    // can we determine?
                    if (!string.IsNullOrEmpty(TrainingPath)) { Mode = RuntimeMode.Training; break; }
                    if (!string.IsNullOrEmpty(RunFile)) { Mode = RuntimeMode.Running; break; }
                    throw new Exception("No valid --mode: specified, and can't infer usage!");
            }

            if (!string.IsNullOrEmpty(loops))
            {
                if (int.TryParse(loops, out var loopVal)) Loops = loopVal;
                else throw new Exception("Invalid --loops: value - must be a valid int!");
            }
            else Loops = 3; // default for training

            if (!string.IsNullOrEmpty(hiddenlayers))
            {
                // means we want a specific size setup
                var tmp = hiddenlayers.Split(',', ':'); // 738:100:64:10 as an example or 738,40,10
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

        private static string? GetIfArg(string[] args, string name)
        {
            string arg = args.FirstOrDefault(p => p.StartsWith($"--{name}:") || p.StartsWith($"--{name}="));
            if (string.IsNullOrEmpty(arg) && args.Any(p => p == "--" + name))
            {
                // try the other form: --name valueHere
                for (int i = 0; i < args.Length - 1; i++)
                    if (args[i] == "--" + name) arg = args[i + 1];
            }
            if (string.IsNullOrEmpty(arg)) return null; // not present
            int idxColon = arg.IndexOf(':');
            int idxEqual = arg.IndexOf('=');
            if (idxColon > -1 && idxEqual > -1)
                return arg.Substring(Math.Min(idxColon, idxEqual) + 1);
            else if (idxColon > -1)
                return arg.Substring(idxColon + 1);
            else if (idxEqual > -1)
                return arg.Substring(idxEqual + 1);
            else
                return null;
        }

        public readonly string? TrainingPath;
        public readonly string? RunFile;
        public readonly string? ModelFile;
        public readonly RuntimeMode Mode;
        public readonly int Loops;
        public readonly int[] RequestedHiddenLayers;

        public enum RuntimeMode
        {
            Training,
            Running,
        }

        // given an index (vector), what is the character?
        public char DevectorizeKey(int idx) => VECTOR_MAPPING.FirstOrDefault(p => p.Key == idx).Value;

        // basicly, turn a character into its index (or vector) for comparison
        public int VectorizeKey(char key) => VECTOR_MAPPING.FirstOrDefault(p => p.Value == key).Key;
    }
}
