// Computational Classnotes — Cart-pole and learned pendulum (Unity 6)
// From "Feedforward Neural Networks: The Algebra and the Computing", Sections 8–9.
// © Computational Classnotes · CC BY-NC-SA 4.0

using System;

namespace CCN.Core
{
    /// <summary>
    /// A small feedforward network: logistic-sigmoid hidden layers, linear output layer.
    /// Weight matrices are stored rows = neurons of the next layer, columns = inputs,
    /// and each layer computes z = W·a + b (the notebook's convention).
    /// The input is multiplied component-wise by InputScale before the first layer.
    /// </summary>
    public sealed class TinyNet
    {
        public readonly double[][,] W;
        public readonly double[][] B;
        public readonly double[] InputScale;

        public int InputSize => W[0].GetLength(1);
        public int OutputSize => W[W.Length - 1].GetLength(0);

        public TinyNet(double[][,] weights, double[][] biases, double[] inputScale)
        {
            if (weights == null || biases == null || weights.Length == 0 || weights.Length != biases.Length)
                throw new ArgumentException("Need one bias vector per weight matrix.");
            for (int k = 0; k < weights.Length; k++)
            {
                if (weights[k].GetLength(0) != biases[k].Length)
                    throw new ArgumentException($"Layer {k}: W has {weights[k].GetLength(0)} rows but b has {biases[k].Length} entries.");
                if (k > 0 && weights[k].GetLength(1) != weights[k - 1].GetLength(0))
                    throw new ArgumentException($"Layer {k}: W has {weights[k].GetLength(1)} columns, previous layer has {weights[k - 1].GetLength(0)} neurons.");
            }
            int n0 = weights[0].GetLength(1);
            if (inputScale == null) { inputScale = new double[n0]; for (int i = 0; i < n0; i++) inputScale[i] = 1.0; }
            if (inputScale.Length != n0) throw new ArgumentException("InputScale length must equal the input size.");
            W = weights; B = biases; InputScale = inputScale;
        }

        public static double Sigmoid(double z) => 1.0 / (1.0 + Math.Exp(-z));

        /// <summary>Deep copy (weights, biases and input scale).</summary>
        public TinyNet Clone()
        {
            var w = new double[W.Length][,];
            var b = new double[B.Length][];
            for (int k = 0; k < W.Length; k++) { w[k] = (double[,])W[k].Clone(); b[k] = (double[])B[k].Clone(); }
            return new TinyNet(w, b, (double[])InputScale.Clone());
        }

        /// <summary>Evaluates the network. Returns a new array with the output layer.</summary>
        public double[] Evaluate(double[] input)
        {
            if (input.Length != InputSize) throw new ArgumentException("Wrong input size.");
            var a = new double[input.Length];
            for (int i = 0; i < a.Length; i++) a[i] = input[i] * InputScale[i];

            for (int k = 0; k < W.Length; k++)
            {
                var w = W[k];
                int rows = w.GetLength(0), cols = w.GetLength(1);
                var z = new double[rows];
                bool last = k == W.Length - 1;
                for (int r = 0; r < rows; r++)
                {
                    double s = B[k][r];
                    for (int c = 0; c < cols; c++) s += w[r, c] * a[c];
                    z[r] = last ? s : Sigmoid(s);
                }
                a = z;
            }
            return a;
        }

        /// <summary>Convenience for single-input, single-output networks.</summary>
        public double Evaluate(double x) => Evaluate(new[] { x })[0];

        // ---------- conversion to and from the flat JSON layout ----------

        public static TinyNet FromJson(NetJson j)
        {
            if (j == null || j.layers == null || j.layers.Length == 0) throw new ArgumentException("Empty network description.");
            var Ws = new double[j.layers.Length][,];
            var Bs = new double[j.layers.Length][];
            for (int k = 0; k < j.layers.Length; k++)
            {
                var L = j.layers[k];
                if (L.W == null || L.W.Length != L.rows * L.cols)
                    throw new ArgumentException($"Layer {k}: W must have rows*cols = {L.rows * L.cols} numbers (row-major).");
                var w = new double[L.rows, L.cols];
                for (int r = 0; r < L.rows; r++)
                    for (int c = 0; c < L.cols; c++)
                        w[r, c] = L.W[r * L.cols + c];
                Ws[k] = w;
                Bs[k] = (double[])L.b.Clone();
            }
            return new TinyNet(Ws, Bs, j.inputScale != null && j.inputScale.Length > 0 ? (double[])j.inputScale.Clone() : null);
        }

        public NetJson ToJson(string name, string note = "")
        {
            var j = new NetJson { name = name, note = note, inputScale = (double[])InputScale.Clone(), layers = new LayerJson[W.Length] };
            for (int k = 0; k < W.Length; k++)
            {
                int rows = W[k].GetLength(0), cols = W[k].GetLength(1);
                var flat = new double[rows * cols];
                for (int r = 0; r < rows; r++) for (int c = 0; c < cols; c++) flat[r * cols + c] = W[k][r, c];
                j.layers[k] = new LayerJson { rows = rows, cols = cols, W = flat, b = (double[])B[k].Clone() };
            }
            return j;
        }
    }

    /// <summary>JSON layout used in StreamingAssets/CCN (works with UnityEngine.JsonUtility).</summary>
    [Serializable]
    public class NetJson
    {
        public string name;
        public string note;
        public double[] inputScale;
        public LayerJson[] layers;
    }

    [Serializable]
    public class LayerJson
    {
        public int rows;
        public int cols;
        public double[] W;   // row-major, rows*cols numbers
        public double[] b;   // rows numbers
    }
}
