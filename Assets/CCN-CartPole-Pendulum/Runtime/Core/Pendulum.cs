// Computational Classnotes — Cart-pole and learned pendulum (Unity 6)
// Pendulum with g/l = 1, leapfrog (velocity Verlet) integrator, step h = 0.05.
// © Computational Classnotes · CC BY-NC-SA 4.0

using System;

namespace CCN.Core
{
    public sealed class Pendulum
    {
        public const double DefaultH = 0.05;

        public readonly string Name;
        /// <summary>Acceleration law a(θ): −sin θ for the true pendulum, the network for the learned one.</summary>
        public readonly Func<double, double> Accel;
        public readonly double H;

        public double Theta, Omega;
        public int StepCount { get; private set; }
        public double Time => StepCount * H;

        /// <summary>Time between the two most recent turning points on the +θ side (NaN until two are seen).</summary>
        public double MeasuredPeriod { get; private set; } = double.NaN;
        double _lastMaxTime = double.NaN;

        public Pendulum(string name, Func<double, double> accel, double theta0 = 2.0, double omega0 = 0.0, double h = DefaultH)
        {
            Name = name; Accel = accel; H = h;
            Reset(theta0, omega0);
        }

        public static Pendulum True(double theta0 = 2.0) => new Pendulum("True", th => -Math.Sin(th), theta0);
        public static Pendulum Learned(TinyNet net, double theta0 = 2.0) => new Pendulum("Network", th => net.Evaluate(th), theta0);

        public void Reset(double theta0 = 2.0, double omega0 = 0.0)
        {
            Theta = theta0; Omega = omega0; StepCount = 0;
            MeasuredPeriod = double.NaN; _lastMaxTime = double.NaN;
        }

        /// <summary>One leapfrog step: ω½ = ω + (h/2)a(θ); θ' = θ + hω½; ω' = ω½ + (h/2)a(θ').</summary>
        public void Step()
        {
            double prevOmega = Omega;
            double wHalf = Omega + 0.5 * H * Accel(Theta);
            Theta += H * wHalf;
            Omega = wHalf + 0.5 * H * Accel(Theta);
            StepCount++;

            // Turning point on the +θ side: ω changes sign from + to −.
            if (prevOmega > 0 && Omega <= 0 && Theta > 0)
            {
                // Linear interpolation of the crossing time inside the step.
                double frac = prevOmega / (prevOmega - Omega);
                double tCross = (StepCount - 1 + frac) * H;
                if (!double.IsNaN(_lastMaxTime)) MeasuredPeriod = tCross - _lastMaxTime;
                _lastMaxTime = tCross;
            }
        }

        /// <summary>
        /// Energy E = ω²/2 + V(θ) with V' = −a. For the true pendulum V = 1 − cos θ.
        /// For any law, V is computed by integrating −a from 0 to θ (Simpson, 64 panels).
        /// </summary>
        public double Energy()
        {
            const int n = 64;
            double hInt = Theta / n, sum = -Accel(0) - Accel(Theta);
            for (int i = 1; i < n; i++) sum += (i % 2 == 1 ? 4 : 2) * -Accel(i * hInt);
            double V = sum * hInt / 3.0;
            return 0.5 * Omega * Omega + V;
        }
    }
}
