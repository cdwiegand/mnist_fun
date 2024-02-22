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
    public class GuessResult
    {
        public int CountedRight = 0;
        public int CountedWrong = 0;
        public int Total => CountedRight + CountedWrong;
        public string PercentStr => Total > 0 ? ((decimal)CountedRight / Total).ToString("P2") : "N/A";
        public void CountRight() => CountedRight += 1;
        public void CountWrong() => CountedWrong += 1;
    }
}