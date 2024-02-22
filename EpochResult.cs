using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mnistfun
{
    public class EpochResult
    {
        public int EpochGeneration;
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

        public override string ToString()
        {
            string ret = $"\nEpoch {EpochGeneration}: {Accuracy:P3}\n";
            int idx = 0;
            foreach (char i in Characters.Keys.OrderBy(p => p))
                ret += (idx++ % 12 == 0 ? "\n" : "") + $" {i}: {Characters[i].PercentStr}";
            return ret;
        }

        public decimal Accuracy => (decimal)CountedCorrect / (decimal)CountedTotal;
    }
}
