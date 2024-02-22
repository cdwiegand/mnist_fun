using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mnistfun
{
    public class Args
    {
        public Args(string[] args)
        {
            TrainingPath = GetIfArg(args, "train");
            RunFile = GetIfArg(args, "run");
            ModelFile = GetIfArg(args, "model");
            string? mode = GetIfArg(args, "mode");

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

            // valid?
            if (string.IsNullOrEmpty(TrainingPath) && string.IsNullOrEmpty(RunFile))
                throw new Exception("Must specify at least a training path or a run file!");
            if (!string.IsNullOrEmpty(RunFile) && (string.IsNullOrEmpty(ModelFile) || !System.IO.File.Exists(ModelFile)))
                throw new Exception("If running, must specify a VALID model file!");
        }

        private static string? GetIfArg(string[] args, string arg) => GetValue(args.FirstOrDefault(p => p.StartsWith($"--{arg}:")));
        private static string? GetValue(string arg) => string.IsNullOrEmpty(arg) ? null : arg.Contains(':') ? arg.Substring(arg.IndexOf(':') + 1) : null;

        public readonly string? TrainingPath;
        public readonly string? RunFile;
        public readonly string? ModelFile;
        public readonly RuntimeMode Mode;

        public enum RuntimeMode
        {
            Training,
            Running,
        }
    }
}
