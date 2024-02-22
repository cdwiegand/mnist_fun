using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json.Nodes;
using System.Threading.Tasks;

namespace mnistfun
{
    public class EpochResult
    {
        public int EpochGeneration;
        public DateTime StartTime;
        public DateTime? EndTime;
        public TimeSpan? Duration => EndTime.HasValue ? EndTime.Value.Subtract(StartTime) : null;

        public Dictionary<char, GuessResult> Characters = new Dictionary<char, GuessResult>();
        public int CountedCorrect => Characters.Select(p => p.Value.CountedRight).Sum();
        public int CountedWrong => Characters.Select(p => p.Value.CountedWrong).Sum();
        public int CountedTotal => Characters.Select(p => p.Value.CountedRight + p.Value.CountedWrong).Sum();

        public void CountRight(char number)
        {
            if (!Characters.ContainsKey(number))
                Characters.Add(number, new GuessResult());
            Characters[number].CountRight();
        }

        public void CountWrong(char number)
        {
            if (!Characters.ContainsKey(number))
                Characters.Add(number, new GuessResult());
            Characters[number].CountWrong();
        }

        public JsonNode ToJson()
        {
            JsonObject root = new JsonObject();
            root["generation"] = EpochGeneration;
            root["accuracy"] = Accuracy;
            if (Duration.HasValue) root["duration"] = Duration.Value.TotalSeconds;
            root["values"] = new JsonArray(Characters.OrderBy(p => p.Key).Select(p => p.Value.ToJson(p.Key)).ToArray());
            return root;
        }

        public override string ToString()
        {
            string ret = $"\nEpoch {EpochGeneration}: {Accuracy:P3} @ {Duration}\n";
            int idx = 0;
            foreach (char i in Characters.Keys.OrderBy(p => p))
                ret += (idx++ % 12 == 0 ? "\n" : "") + $" {i}: {Characters[i].PercentStr}";
            return ret;
        }

        public decimal Accuracy => (decimal)CountedCorrect / (decimal)CountedTotal;
    }
}
