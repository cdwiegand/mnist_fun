using CsvHelper;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System.Net.WebSockets;
using System.Reflection;
using System.Reflection.Metadata;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json.Nodes;

namespace mnistfun
{
    public class GuessResult
    {
        public int CountedRight = 0;
        public int CountedWrong = 0;
        public int Total => CountedRight + CountedWrong;
        public decimal? Percent => Total > 0 ? ((decimal)CountedRight / Total) : null;
        public string PercentStr => Percent?.ToString("P2") ?? "N/A";
        public void CountRight() => CountedRight += 1;
        public void CountWrong() => CountedWrong += 1;

        public JsonNode ToJson(char c)
        {
            JsonObject ret = new JsonObject();
            ret["key"] = c;
            ret["count_total"] = Total;
            ret["count_right"] = CountedRight;
            ret["count_wrong"] = CountedWrong;
            ret["percent"] = Percent;
            return ret;
        }
    }
}