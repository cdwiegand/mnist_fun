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
    public class SourceData : List<Tuple<char, double[]>>
    {
        public SourceData() { }
        public SourceData(IEnumerable<Tuple<char, double[]>> src) => this.AddRange(src);
    }
}