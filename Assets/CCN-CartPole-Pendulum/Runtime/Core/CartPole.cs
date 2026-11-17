// Computational Classnotes — Cart-pole and learned pendulum (Unity 6)
// Cart-pole physics (Barto, Sutton and Anderson 1983), frictionless, RK4 with dt = 0.02 s.
// © Computational Classnotes · CC BY-NC-SA 4.0

using System;

namespace CCN.Core
{
    /// <summary>State s = (x, v, θ, ω): cart position, cart speed, angle from upright (+ tips towards +x), angular speed.</summary>
    public struct CartPoleState
    {
        public double X, V, Theta, Omega;
        public CartPoleState(double x, double v, double theta, double omega) { X = x; V = v; Theta = theta; Omega = omega; }
        public double[] ToArray() => new[] { X, V, Theta, Omega };
        public override string ToString() => $"(x={X:F4}, v={V:F4}, θ={Theta:F4}, ω={Omega:F4})";
    }

    public sealed class CartPole
    {
        public const double G = 9.81;      // m/s²
        public const double Mc = 1.0;      // cart mass, kg
        public const double Mp = 0.1;      // stick mass, kg
        public const double L = 0.5;       // HALF the stick length, m
        public const double UMax = 10.0;   // N
        public const double Dt = 0.02;     // s

        public CartPoleState State;
        public bool Fallen { get; private set; }
        public int StepCount { get; private set; }
        public double Time => StepCount * Dt;
        /// <summary>Force applied during the most recent step (N).</summary>
        public double LastForce { get; private set; }

        public CartPole(CartPoleState s0) { Reset(s0); }

        public void Reset(CartPoleState s0)
        {
            State = s0; Fallen = Math.Abs(s0.Theta) > Math.PI / 2; StepCount = 0; LastForce = 0;
        }

        /// <summary>Time derivative of the state for a constant force u.</summary>
        public static CartPoleState Derivative(CartPoleState s, double u)
        {
            double sin = Math.Sin(s.Theta), cos = Math.Cos(s.Theta);
            double total = Mc + Mp;
            double f = (u + Mp * L * s.Omega * s.Omega * sin) / total;
            double dOmega = (G * sin - cos * f) / (L * (4.0 / 3.0 - Mp * cos * cos / total));
            double dV = f - Mp * L * dOmega * cos / total;
            return new CartPoleState(s.V, dV, s.Omega, dOmega);
        }

        static CartPoleState Add(CartPoleState a, CartPoleState k, double h) =>
            new CartPoleState(a.X + h * k.X, a.V + h * k.V, a.Theta + h * k.Theta, a.Omega + h * k.Omega);

        /// <summary>One classic RK4 step of length dt with u held constant.</summary>
        public static CartPoleState Rk4(CartPoleState s, double u, double dt)
        {
            var k1 = Derivative(s, u);
            var k2 = Derivative(Add(s, k1, dt / 2), u);
            var k3 = Derivative(Add(s, k2, dt / 2), u);
            var k4 = Derivative(Add(s, k3, dt), u);
            return new CartPoleState(
                s.X + dt / 6 * (k1.X + 2 * k2.X + 2 * k3.X + k4.X),
                s.V + dt / 6 * (k1.V + 2 * k2.V + 2 * k3.V + k4.V),
                s.Theta + dt / 6 * (k1.Theta + 2 * k2.Theta + 2 * k3.Theta + k4.Theta),
                s.Omega + dt / 6 * (k1.Omega + 2 * k2.Omega + 2 * k3.Omega + k4.Omega));
        }

        /// <summary>
        /// Advances one step: the controller computes u from the current state, then RK4 runs with u fixed.
        /// Once |θ| > π/2 the stick has fallen and the state is frozen (as in the notebook).
        /// </summary>
        public void Step(ICartPoleController controller)
        {
            if (Fallen) return;
            double u = controller?.Force(State) ?? 0.0;
            LastForce = u;
            State = Rk4(State, u, Dt);
            StepCount++;
            if (Math.Abs(State.Theta) > Math.PI / 2) Fallen = true;
        }

        /// <summary>A sudden hit on the stick: adds Δω (rad/s).</summary>
        public void Kick(double deltaOmega)
        {
            if (!Fallen) State.Omega += deltaOmega;
        }
    }
}
