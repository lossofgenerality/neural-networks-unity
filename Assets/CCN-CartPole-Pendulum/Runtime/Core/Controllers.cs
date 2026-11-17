// Computational Classnotes — Cart-pole and learned pendulum (Unity 6)
// Cart-pole controllers: trained 4-8-1 network, LQR teacher with force limit, and Off.
// © Computational Classnotes · CC BY-NC-SA 4.0

using System;

namespace CCN.Core
{
    public interface ICartPoleController
    {
        string Name { get; }
        /// <summary>Force u (N) for the state s.</summary>
        double Force(CartPoleState s);
    }

    /// <summary>The teacher: u = u_max · tanh(−(K·s)/u_max).</summary>
    public sealed class LqrController : ICartPoleController
    {
        public readonly double[] K;
        public LqrController(double[] k = null) { K = k ?? TrainedWeights.LqrK; }
        public string Name => "LQR";
        public double Force(CartPoleState s)
        {
            double ks = K[0] * s.X + K[1] * s.V + K[2] * s.Theta + K[3] * s.Omega;
            return CartPole.UMax * Math.Tanh(-ks / CartPole.UMax);
        }
    }

    /// <summary>The trained network: u = clamp(10 · net(s), −10, 10).</summary>
    public sealed class NetworkController : ICartPoleController
    {
        public readonly TinyNet Net;
        public NetworkController(TinyNet net = null) { Net = net ?? TrainedWeights.CartPoleNet(); }
        public string Name => "Network";
        public double Force(CartPoleState s)
        {
            double o = Net.Evaluate(s.ToArray())[0];
            double u = CartPole.UMax * o;
            return Math.Max(-CartPole.UMax, Math.Min(CartPole.UMax, u));
        }
    }

    /// <summary>No control: u = 0.</summary>
    public sealed class OffController : ICartPoleController
    {
        public string Name => "Off";
        public double Force(CartPoleState s) => 0.0;
    }
}
