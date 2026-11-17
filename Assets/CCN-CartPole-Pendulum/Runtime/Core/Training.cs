// Computational Classnotes — Cart-pole and learned pendulum (Unity 6)
// Native training: full-batch backpropagation with plain gradient descent or Adam,
// following the notebook's recipes (Sections 8–9). Pure C#, safe to run on a worker thread.
// © Computational Classnotes · CC BY-NC-SA 4.0

using System;

namespace CCN.Core
{
    public sealed class TrainingData
    {
        public double[][] X;   // raw inputs (the network applies its own InputScale)
        public double[][] Y;   // targets
        public int Count => X.Length;
    }

    public enum OptimizerKind { GradientDescent, Adam }

    public sealed class TrainerConfig
    {
        public OptimizerKind Optimizer = OptimizerKind.GradientDescent;
        public double LearningRate = 0.5;
        public int Epochs = 4000;
        public double Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-8;
        /// <summary>Record the loss every this many epochs (and at the last epoch).</summary>
        public int LogEvery = 10;
        /// <summary>Epochs after which a copy of the network is kept (0 = the untrained start).</summary>
        public int[] SnapshotEpochs = { 0, 20, 40, 100, 4000 };
    }

    /// <summary>The two training recipes from the notebook.</summary>
    public static class Recipes
    {
        /// <summary>Normal(0, 1) by Box–Muller.</summary>
        public static double Gaussian(Random r)
        {
            double u1 = 1.0 - r.NextDouble(), u2 = r.NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
        }

        /// <summary>Weights normal with sd 1/√(fan-in), zero biases.</summary>
        public static TinyNet RandomNet(int[] sizes, double[] inputScale, int seed)
        {
            var r = new Random(seed);
            int L = sizes.Length - 1;
            var W = new double[L][,];
            var B = new double[L][];
            for (int k = 0; k < L; k++)
            {
                int rows = sizes[k + 1], cols = sizes[k];
                double sd = 1.0 / Math.Sqrt(cols);
                W[k] = new double[rows, cols];
                for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) W[k][i, j] = sd * Gaussian(r);
                B[k] = new double[rows];
            }
            return new TinyNet(W, B, (double[])inputScale.Clone());
        }

        // ---- cart-pole: imitate the LQR teacher ----
        public static readonly double[] CartPoleScale = { 1.0, 0.5, 2.0, 0.5 };

        /// <summary>3000 states uniform in x∈[−1,1], v∈[−2,2], θ∈[−0.5,0.5], ω∈[−2,2]; target = teacher(s)/10.</summary>
        public static TrainingData CartPoleData(int n = 3000, int seed = 3)
        {
            var r = new Random(seed);
            var teacher = new LqrController();
            var d = new TrainingData { X = new double[n][], Y = new double[n][] };
            for (int i = 0; i < n; i++)
            {
                var s = new CartPoleState(r.NextDouble() * 2 - 1, r.NextDouble() * 4 - 2, r.NextDouble() - 0.5, r.NextDouble() * 4 - 2);
                d.X[i] = s.ToArray();
                d.Y[i] = new[] { teacher.Force(s) / CartPole.UMax };
            }
            return d;
        }

        public static TinyNet CartPoleStart(int seed = 5) => RandomNet(new[] { 4, 8, 1 }, CartPoleScale, seed);

        public static TrainerConfig CartPoleConfig() => new TrainerConfig
        {
            Optimizer = OptimizerKind.GradientDescent, LearningRate = 0.5, Epochs = 4000, LogEvery = 10,
            SnapshotEpochs = new[] { 0, 20, 40, 100, 200, 500, 1000, 2000, 4000 },
        };

        // ---- pendulum: learn a(θ) = −sin θ from noisy samples ----
        public static readonly double[] PendulumScale = { 0.5 };

        /// <summary>200 angles uniform in [−2.6, 2.6]; targets −sin θ plus normal noise with sd 0.02.</summary>
        public static TrainingData PendulumData(int n = 200, int seed = 11, double noise = 0.02)
        {
            var r = new Random(seed);
            var d = new TrainingData { X = new double[n][], Y = new double[n][] };
            for (int i = 0; i < n; i++)
            {
                double th = r.NextDouble() * 5.2 - 2.6;
                d.X[i] = new[] { th };
                d.Y[i] = new[] { -Math.Sin(th) + noise * Gaussian(r) };
            }
            return d;
        }

        public static TinyNet PendulumStart(int seed = 1) => RandomNet(new[] { 1, 6, 1 }, PendulumScale, seed);

        public static TrainerConfig PendulumConfig() => new TrainerConfig
        {
            Optimizer = OptimizerKind.Adam, LearningRate = 0.03, Epochs = 2000, LogEvery = 5,
            SnapshotEpochs = new[] { 0, 25, 50, 100, 200, 500, 1000, 2000 },
        };
    }

    /// <summary>
    /// Full-batch trainer for sigmoid-hidden / linear-output networks.
    /// Loss = mean over samples of the squared error, L = (1/N) Σ |out − y|².
    /// </summary>
    public sealed class Trainer
    {
        public readonly TinyNet Net;          // trained in place
        public readonly TrainerConfig Config;
        readonly TrainingData _data;

        // gradient and Adam buffers, same shapes as the weights
        readonly double[][,] _gW, _mW, _vW;
        readonly double[][] _gB, _mB, _vB;
        // forward-pass storage: activations a[k] (a[0] = scaled input) for one sample
        readonly double[][] _a, _delta;

        public int Epoch { get; private set; }

        public Trainer(TinyNet start, TrainingData data, TrainerConfig config)
        {
            Net = start.Clone(); _data = data; Config = config;
            int L = Net.W.Length;
            _gW = new double[L][,]; _mW = new double[L][,]; _vW = new double[L][,];
            _gB = new double[L][]; _mB = new double[L][]; _vB = new double[L][];
            _a = new double[L + 1][]; _delta = new double[L][];
            _a[0] = new double[Net.InputSize];
            for (int k = 0; k < L; k++)
            {
                int rows = Net.W[k].GetLength(0), cols = Net.W[k].GetLength(1);
                _gW[k] = new double[rows, cols]; _mW[k] = new double[rows, cols]; _vW[k] = new double[rows, cols];
                _gB[k] = new double[rows]; _mB[k] = new double[rows]; _vB[k] = new double[rows];
                _a[k + 1] = new double[rows]; _delta[k] = new double[rows];
            }
        }

        void Forward(double[] x)
        {
            var a0 = _a[0];
            for (int i = 0; i < a0.Length; i++) a0[i] = x[i] * Net.InputScale[i];
            int L = Net.W.Length;
            for (int k = 0; k < L; k++)
            {
                var w = Net.W[k]; var b = Net.B[k]; var ain = _a[k]; var aout = _a[k + 1];
                int rows = w.GetLength(0), cols = w.GetLength(1);
                bool last = k == L - 1;
                for (int r = 0; r < rows; r++)
                {
                    double s = b[r];
                    for (int c = 0; c < cols; c++) s += w[r, c] * ain[c];
                    aout[r] = last ? s : TinyNet.Sigmoid(s);
                }
            }
        }

        /// <summary>Mean squared error of the current network on the training data.</summary>
        public double Loss()
        {
            double sum = 0; var outp = _a[_a.Length - 1];
            for (int n = 0; n < _data.Count; n++)
            {
                Forward(_data.X[n]);
                for (int o = 0; o < outp.Length; o++) { double e = outp[o] - _data.Y[n][o]; sum += e * e; }
            }
            return sum / _data.Count;
        }

        /// <summary>One full-batch update. Returns the loss before the update.</summary>
        public double Step()
        {
            int L = Net.W.Length, N = _data.Count;
            for (int k = 0; k < L; k++) { Array.Clear(_gW[k], 0, _gW[k].Length); Array.Clear(_gB[k], 0, _gB[k].Length); }

            double loss = 0;
            for (int n = 0; n < N; n++)
            {
                Forward(_data.X[n]);
                // output layer (linear): dL/dz = 2 (out − y) / N
                var outp = _a[L]; var dl = _delta[L - 1];
                for (int o = 0; o < outp.Length; o++)
                {
                    double e = outp[o] - _data.Y[n][o];
                    loss += e * e;
                    dl[o] = 2.0 * e / N;
                }
                for (int k = L - 1; k >= 0; k--)
                {
                    var d = _delta[k]; var ain = _a[k]; var gw = _gW[k]; var gb = _gB[k];
                    int rows = gw.GetLength(0), cols = gw.GetLength(1);
                    for (int r = 0; r < rows; r++)
                    {
                        gb[r] += d[r];
                        for (int c = 0; c < cols; c++) gw[r, c] += d[r] * ain[c];
                    }
                    if (k > 0)
                    {
                        // back through W_k, then through the sigmoid of layer k−1: σ' = a(1 − a)
                        var w = Net.W[k]; var dPrev = _delta[k - 1];
                        for (int c = 0; c < cols; c++)
                        {
                            double s = 0;
                            for (int r = 0; r < rows; r++) s += w[r, c] * d[r];
                            double a = ain[c];
                            dPrev[c] = s * a * (1 - a);
                        }
                    }
                }
            }

            Epoch++;
            double lr = Config.LearningRate;
            if (Config.Optimizer == OptimizerKind.GradientDescent)
            {
                for (int k = 0; k < L; k++)
                {
                    var w = Net.W[k]; var gw = _gW[k]; var b = Net.B[k]; var gb = _gB[k];
                    int rows = w.GetLength(0), cols = w.GetLength(1);
                    for (int r = 0; r < rows; r++)
                    {
                        b[r] -= lr * gb[r];
                        for (int c = 0; c < cols; c++) w[r, c] -= lr * gw[r, c];
                    }
                }
            }
            else
            {
                double b1 = Config.Beta1, b2 = Config.Beta2, eps = Config.Epsilon;
                double c1 = 1 - Math.Pow(b1, Epoch), c2 = 1 - Math.Pow(b2, Epoch);
                for (int k = 0; k < L; k++)
                {
                    var w = Net.W[k]; var b = Net.B[k];
                    int rows = w.GetLength(0), cols = w.GetLength(1);
                    for (int r = 0; r < rows; r++)
                    {
                        double g = _gB[k][r];
                        _mB[k][r] = b1 * _mB[k][r] + (1 - b1) * g;
                        _vB[k][r] = b2 * _vB[k][r] + (1 - b2) * g * g;
                        b[r] -= lr * (_mB[k][r] / c1) / (Math.Sqrt(_vB[k][r] / c2) + eps);
                        for (int c = 0; c < cols; c++)
                        {
                            g = _gW[k][r, c];
                            _mW[k][r, c] = b1 * _mW[k][r, c] + (1 - b1) * g;
                            _vW[k][r, c] = b2 * _vW[k][r, c] + (1 - b2) * g * g;
                            w[r, c] -= lr * (_mW[k][r, c] / c1) / (Math.Sqrt(_vW[k][r, c] / c2) + eps);
                        }
                    }
                }
            }
            return loss / N;
        }

        /// <summary>
        /// Runs all epochs. onLog(epoch, loss) gets the loss every LogEvery epochs and at the end;
        /// onSnapshot(epoch, copy) gets a copy of the network at each snapshot epoch; cancel() stops early.
        /// </summary>
        public void Run(Action<int, double> onLog, Action<int, TinyNet> onSnapshot, Func<bool> cancel)
        {
            var snaps = Config.SnapshotEpochs ?? new int[0];
            if (Array.IndexOf(snaps, 0) >= 0) onSnapshot?.Invoke(0, Net.Clone());
            onLog?.Invoke(0, Loss());
            while (Epoch < Config.Epochs)
            {
                if (cancel != null && cancel()) return;
                Step();
                if (Array.IndexOf(snaps, Epoch) >= 0) onSnapshot?.Invoke(Epoch, Net.Clone());
                if (Epoch % Config.LogEvery == 0 || Epoch == Config.Epochs) onLog?.Invoke(Epoch, Loss());
            }
        }
    }
}
