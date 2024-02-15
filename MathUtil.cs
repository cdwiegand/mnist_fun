using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace mnistfun
{
    public static class MathUtil
    {
        public static double[] MatrixMultiply(double[,] data, double[] columnMultipliers)
        {
            if (columnMultipliers.Length != data.GetLength(1)) throw new Exception("Bad matrix!");

            double[] ret = new double[data.GetLength(0)];
            FillWithValue(0, ret); // initialize to zero

            for (int i = 0; i < data.GetLength(0); i++)
                for (int j = 0; j < data.GetLength(1); j++)
                    ret[i] += data[i, j] * columnMultipliers[j];
            return ret;
        }

        public static double Sigmoid(double h_pre) => (double)(1 / (1 + Math.Pow(Math.E, (double)(-h_pre))));

        public static void FillWithRandom(Random rand, double[] matrix)
        {
            for (int i2hl1 = 0; i2hl1 < matrix.GetLength(0); i2hl1++)
                matrix[i2hl1] = rand.NextDouble() - 0.5;
        }
        public static void FillWithValue(double value, double[] matrix)
        {
            for (int i2hl1 = 0; i2hl1 < matrix.GetLength(0); i2hl1++)
                matrix[i2hl1] = value;
        }
        public static void FillWithRandom(Random rand, double[,] matrix)
        {
            for (int i = 0; i < matrix.GetLength(0); i++)
                for (int j = 0; j < matrix.GetLength(1); j++)
                    matrix[i, j] = rand.NextDouble() - 0.5;
        }
        public static void FillWithValue(double value, double[,] matrix)
        {
            for (int i = 0; i < matrix.GetLength(0); i++)
                for (int j = 0; j < matrix.GetLength(1); j++)
                    matrix[i, j] = value;
        }
        public static void FillWithValue(int value, int[] matrix)
        {
            for (int i2hl1 = 0; i2hl1 < matrix.GetLength(0); i2hl1++)
                matrix[i2hl1] = value;
        }
    }
}
