using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace mnistfun
{
    [Serializable]
    public class InterLayerMatrix
    {
        public InterLayerMatrix(Layer fromLayer, Layer toLayer)
        {
            FromLayer = fromLayer;
            ToLayer = toLayer;
            Matrix = MathNet.Numerics.LinearAlgebra.Matrix<double>.Build.Random(fromLayer.Length, toLayer.Length).ToArray();
        }

        public double[,] Matrix { get; set; }

        public JsonNode ToJson()
        {
            JsonObject root = new JsonObject();
            root.Add("FromLayerId", FromLayer.Id);
            root.Add("ToLayerId", ToLayer.Id);

            // system.text.json also can't double[,] -> json.. why I'm not using Newtonsoft I don't know...
            JsonArray jsonMatrix = new JsonArray();
            for (int x = 0; x < Matrix.GetLength(0); x++)
            {
                JsonArray row = new JsonArray();
                for (int y = 0; y < Matrix.GetLength(1); y++)
                    row.Add(Matrix[x, y]);
                jsonMatrix.Add(row);
            }
            root.Add("Matrix", jsonMatrix);
            return root;
        }
        public static InterLayerMatrix FromJson(JsonNode json, Model model)
        {
            Guid FromLayerId = json["FromLayerId"].Deserialize<Guid>();
            Guid ToLayerId = json["ToLayerId"].Deserialize<Guid>();
            var ret = new InterLayerMatrix(model.GetLayer(FromLayerId), model.GetLayer(ToLayerId));

            JsonArray jsonMatrix = json["Matrix"].AsArray();
            int xCount = jsonMatrix.Count;
            JsonArray firstRow = jsonMatrix[0].AsArray();
            int yCount = firstRow.Count; // yes, I know, it's not really first "row", it's more first column, just so long as it's consistent with ToJson() it doesn't matter
            ret.Matrix = new double[xCount, yCount];
            for (int x = 0; x < xCount; x++)
            {
                JsonArray row = jsonMatrix[x].AsArray();
                for (int y = 0; y < row.Count; y++)
                    ret.Matrix[x, y] = row[y].AsValue().GetValue<double>();
            }
            return ret;
        }

        public readonly Layer FromLayer;
        public readonly Layer ToLayer;

        public void ApplyMatricesForward()
        {
            // weight matrix 1st dimension is left/input layer (rows), 2nd dimension is right/output layer (column)
            // (note: opposite of youtube video I'm watching!)
            int columns = Matrix.GetLength(1);
            ToLayer.Reset();
            for (int resultIdx = 0; resultIdx < columns; resultIdx++)
            {
                double h_pre = 0;
                for (int pix = 0; pix < Matrix.GetLength(0); pix++)
                    // multiply them by the left -> right layer, then apply sigmoid function to that                            
                    h_pre += FromLayer.Neurons[pix] * Matrix[pix, resultIdx];
                ToLayer.Neurons[resultIdx] = MathUtil.Sigmoid(h_pre);
            }
        }

        public double[] CalculateBackpropagationMatrixDelta(double[] deltaOutput)
        {
            double[] deltaHiddenLayer1a = new double[FromLayer.Length];
            for (int i = 0; i < FromLayer.Length; i++)
                deltaHiddenLayer1a[i] = (FromLayer.Neurons[i] * (1 - FromLayer.Neurons[i]));
            double[] deltaHiddenLayer1b = MathUtil.MatrixMultiply(Matrix, deltaOutput);
            double[] deltaHiddenLayer1c = new double[FromLayer.Length];
            for (int i = 0; i < deltaHiddenLayer1c.Length; i++)
                deltaHiddenLayer1c[i] = deltaHiddenLayer1b[i] * deltaHiddenLayer1a[i];
            return deltaHiddenLayer1c;
        }

        public void ApplyBackPropagationMatrix(double[,] correctionMatrix)
        {
            // should be the same shape (dimensions) so easy!
            for (int i = 0; i < correctionMatrix.GetLength(0); i++)
                for (int j = 0; j < correctionMatrix.GetLength(1); j++)
                    Matrix[i, j] += correctionMatrix[i, j];
        }
    }
}
