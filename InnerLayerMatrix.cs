using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mnistfun
{
    public class InterLayerMatrix
    {
        public InterLayerMatrix(Layer fromLayer, Layer toLayer)
        {
            FromLayer = fromLayer;
            ToLayer = toLayer;
            matrix = new double[fromLayer.neurons.Length, toLayer.neurons.Length];

            MathUtil.FillWithRandom(Program.rand, matrix);
        }
        public double[,] matrix;
        public readonly Layer FromLayer;
        public readonly Layer ToLayer;

        public void ApplyMatricesForward()
        {
            // weight matrix 1st dimension is left/input layer (rows), 2nd dimension is right/output layer (column)
            // (note: opposite of youtube video I'm watching!)
            int columns = matrix.GetLength(1);
            ToLayer.Reset();
            for (int resultIdx = 0; resultIdx < columns; resultIdx++)
            {
                double h_pre = 0;
                for (int pix = 0; pix < matrix.GetLength(0); pix++)
                    // multiply them by the input -> hidden layer 1, then apply sigmoid function to that                            
                    h_pre += FromLayer.neurons[pix] * matrix[pix, resultIdx];
                ToLayer.neurons[resultIdx] = MathUtil.Sigmoid(h_pre);
            }
        }

        public double[] CalculateBackpropagationMatrixDelta(double[] deltaOutput)
        {
            double[] deltaHiddenLayer1a = new double[FromLayer.Length];
            for (int i = 0; i < FromLayer.Length; i++)
                deltaHiddenLayer1a[i] = (FromLayer.neurons[i] * (1 - FromLayer.neurons[i]));
            double[] deltaHiddenLayer1b = MathUtil.MatrixMultiply(matrix, deltaOutput);
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
                    matrix[i, j] += correctionMatrix[i, j];
        }
    }
}
