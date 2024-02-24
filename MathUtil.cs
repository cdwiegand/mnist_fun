using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Complex;
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
            MathNet.Numerics.LinearAlgebra.Matrix<double> A = Matrix<double>.Build.DenseOfArray(data);
            MathNet.Numerics.LinearAlgebra.Vector<double> B = Vector<double>.Build.DenseOfArray(columnMultipliers);
            return (A * B).ToArray();
        }

        public static double Sigmoid(double h_pre) => (double)(1 / (1 + Math.Pow(Math.E, (double)(-h_pre))));
             
        public static Dictionary<int, double> InitializeSingleMatrix(double countNeurons) 
        {
            Dictionary<int, double> matrix = new Dictionary<int, double>();
            for (int i = 0; i < countNeurons; i++) matrix[i] = 0;
            return matrix;
        }
        public static void FillWithRandom(Random rand, Dictionary<int, double> matrix)
        {
            foreach (int key in matrix.Keys) matrix[key] = rand.NextDouble() - 0.5;
        }
    }
}
