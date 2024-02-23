using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json.Nodes;
using System.Threading.Tasks;

namespace mnistfun
{
    public class TrainingLoopResult
    {
        public int LoopGeneration { get; set; }
        public DateTime StartTime { get; set; }
        public DateTime? EndTime { get; set; }
        public TimeSpan? Duration => EndTime.HasValue ? EndTime.Value.Subtract(StartTime) : null;

        public Dictionary<char, GuessResult> Characters { get; set; } = new Dictionary<char, GuessResult>();
        public int CountedCorrect => Characters.Select(p => p.Value.CountedRight).Sum();
        public int CountedWrong => Characters.Select(p => p.Value.CountedWrong).Sum();
        public int CountedTotal => Characters.Select(p => p.Value.CountedRight + p.Value.CountedWrong).Sum();
        public decimal Accuracy => (decimal)CountedCorrect / (decimal)CountedTotal;

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

        public static TrainingLoopResult FromJson(JsonNode node) => System.Text.Json.JsonSerializer.Deserialize<TrainingLoopResult>(node, Program.DefaultJsonSerializeOptions);

        public JsonNode ToJson() => System.Text.Json.JsonSerializer.Serialize(this, Program.DefaultJsonSerializeOptions);

        public override string ToString()
        {
            string ret = $"\nEpoch {LoopGeneration}: {Accuracy:P3} @ {Duration}\n";
            int idx = 0;
            foreach (char i in Characters.Keys.OrderBy(p => p))
                ret += (idx++ % 12 == 0 ? "\n" : "") + $" {i}: {Characters[i].PercentStr}";
            return ret;
        }
    }
}
