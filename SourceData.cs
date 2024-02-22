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
    public class DataItem
    {
        public char Character { get; set; }
        public double[] Pixels { get; set; }
        public string LoadPath { get; set; }
    }
    public class SourceData : List<DataItem>
    {
        public SourceData() { }
        public SourceData(IEnumerable<DataItem> src) => this.AddRange(src);

        public static SourceData LoadTrainingSource(string path, RuntimeConfig config, SourceData? appendTo = null)
        {
            SourceData sourceData = appendTo ?? new SourceData();

            if (System.IO.Directory.Exists(path))
            {
                foreach (var file in System.IO.Directory.GetDirectories(path))
                    LoadTrainingSource(file, config, sourceData); // recurse
                foreach (var file in System.IO.Directory.GetFiles(path))
                    LoadTrainingSource(file, config, sourceData);
            }
            else if (System.IO.File.Exists(path))
            {
                switch (System.IO.Path.GetExtension(path).Trim().ToLower().Replace(".", ""))
                {
                    case "csv": sourceData = LoadCsvFile(path, config); break;
                    case "png": sourceData.Add(LoadTrainingPngImage(path, new FileInfo(path).Directory.Name[0])); break; // ugly hack but works
                    default: Console.WriteLine("W: Ignoring file " + path); break;
                }
            }

            return sourceData;
        }

        public static SourceData LoadRunningSource(string path, RuntimeConfig config, SourceData? appendTo = null)
        {
            SourceData sourceData = appendTo ?? new SourceData();

            if (System.IO.Directory.Exists(path))
            {
                foreach (var file in System.IO.Directory.GetDirectories(path))
                    LoadRunningSource(file, config, sourceData); // recurse
                foreach (var file in System.IO.Directory.GetFiles(path))
                    LoadRunningSource(file, config, sourceData);
            }
            else if (System.IO.File.Exists(path))
            {
                switch (System.IO.Path.GetExtension(path).Trim().ToLower().Replace(".", ""))
                {
                    case "csv": sourceData.AddRange(LoadCsvFile(path, config)); break;
                    case "png": sourceData.Add(LoadRuntimePngImage(path)); break;
                    default: Console.WriteLine("W: Ignoring file " + path); break;
                }
            }

            return sourceData;
        }

        private static SourceData LoadCsvFile(string loadPath, RuntimeConfig config)
        {
            // path to mnist csv or emnist csv
            var tempData = new SourceData();
            int line = 0;
            using (var reader = new System.IO.StreamReader(loadPath))
            using (var csver = new CsvHelper.CsvReader(reader, System.Globalization.CultureInfo.InvariantCulture, false))
                while (csver.Read()) // no header!
                {
                    line++;
                    int fieldMap = csver.GetField<int>(0);
                    int[] pixelsInt = new int[csver.ColumnCount - 1];
                    for (int i = 1; i < csver.ColumnCount; i++)
                        pixelsInt[i - 1] = csver.GetField<int>(i);
                    double[] pixels = pixelsInt.Select(p => (double)p / 256).ToArray();
                    tempData.Add(new DataItem { Character = config.DevectorizeKey(fieldMap), Pixels = pixels, LoadPath = $"{loadPath}:{line}" });
                }

            return tempData;
        }

        private static string FindPath(Func<string, bool> Test)
        {
            string dirPath = System.Environment.CurrentDirectory;
            while (dirPath != null && !Test(dirPath) && System.IO.Path.GetPathRoot(dirPath) != dirPath)
                dirPath = System.IO.Directory.GetParent(dirPath)?.FullName;
            return dirPath;
        }

        private static SourceData LoadPngImages(string dirPath)
        {
            var mnist_data = new SourceData();
            foreach (var subdir in System.IO.Directory.GetDirectories(dirPath))
            {
                var files = System.IO.Directory.GetFiles(subdir);
                char c = System.IO.Path.GetDirectoryName(subdir)[0];
                foreach (string file in files)
                    mnist_data.Add(LoadTrainingPngImage(file, c));
            }
            return mnist_data;
        }
        private static DataItem LoadTrainingPngImage(string file, char c)
        {
            var image = SixLabors.ImageSharp.Image.Load<SixLabors.ImageSharp.PixelFormats.Argb32>(file);
            double[] pixels = new double[image.Width * image.Height];
            int idx = 0;
            for (int y = 0; y < image.Height; y++)
                for (int x = 0; x < image.Width; x++)
                {
                    double pixel = ((double)image[x, y].R) / 256;
                    pixels[idx++] = pixel; // assign and more to next pixel
                }
            return new DataItem { Character = c, Pixels = pixels, LoadPath = file };
        }
        private static DataItem LoadRuntimePngImage(string file)
        {
            var image = SixLabors.ImageSharp.Image.Load<SixLabors.ImageSharp.PixelFormats.Argb32>(file);
            double[] pixels = new double[image.Width * image.Height];
            int idx = 0;
            for (int y = 0; y < image.Height; y++)
                for (int x = 0; x < image.Width; x++)
                {
                    double pixel = ((double)image[x, y].R) / 256;
                    pixels[idx++] = pixel; // assign and more to next pixel
                }
            return new DataItem { Pixels = pixels, LoadPath = file };
        }
    }
}