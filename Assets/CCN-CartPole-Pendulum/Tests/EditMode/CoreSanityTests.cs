// Computational Classnotes — Cart-pole and learned pendulum (Unity 6)
// Edit-mode tests: the sanity checks from the handoff doc (Window ▸ General ▸ Test Runner ▸ EditMode).
// © Computational Classnotes · CC BY-NC-SA 4.0

using System;
using NUnit.Framework;

namespace CCN.Core.Tests
{
    public class CartPoleTests
    {
        static readonly CartPoleState S0 = new CartPoleState(0, 0, 0.3, 0);

        static CartPole Run(ICartPoleController c, CartPoleState s0, int steps, Action<CartPole, int> beforeStep = null)
        {
            var cp = new CartPole(s0);
            for (int i = 0; i < steps; i++) { beforeStep?.Invoke(cp, i); cp.Step(c); }
            return cp;
        }

        [Test]
        public void Network_KeepsStickUp_For6s_FromNotebookStart()
        {
            var cp = Run(new NetworkController(), S0, 300);
            Assert.IsFalse(cp.Fallen, "stick fell");
            Assert.Less(Math.Abs(cp.State.Theta), 0.05, "should come back near upright");
        }

        [Test]
        public void NoControl_FallsInUnderOneSecond()
        {
            var cp = new CartPole(S0);
            var off = new OffController();
            int steps = 0;
            while (!cp.Fallen && steps < 300) { cp.Step(off); steps++; }
            Assert.IsTrue(cp.Fallen);
            Assert.Less(steps * CartPole.Dt, 1.0);
        }

        [Test]
        public void Lqr_KeepsStickUp()
        {
            var cp = Run(new LqrController(), S0, 300);
            Assert.IsFalse(cp.Fallen);
            Assert.Less(Math.Abs(cp.State.Theta), 0.05);
        }

        [Test]
        public void Network_RecoversFromNotebookPushTest()
        {
            double maxX = 0;
            var cp = new CartPole(new CartPoleState(0, 0, 0.05, 0));
            var net = new NetworkController();
            for (int i = 0; i < 400; i++)
            {
                if (i == 75) cp.Kick(+2);
                if (i == 200) cp.Kick(-2);
                cp.Step(net);
                maxX = Math.Max(maxX, Math.Abs(cp.State.X));
            }
            Assert.IsFalse(cp.Fallen);
            Assert.Less(maxX, 1.6, "cart should stay within about ±1.4 m");
        }

        [Test]
        public void Force_IsLimitedTo10N()
        {
            var big = new CartPoleState(0, 0, 0.5, 2);
            Assert.LessOrEqual(Math.Abs(new NetworkController().Force(big)), 10.0);
            Assert.LessOrEqual(Math.Abs(new LqrController().Force(big)), 10.0);
        }
    }

    public class PendulumTests
    {
        [Test]
        public void Network_ApproximatesMinusSin()
        {
            var net = TrainedWeights.PendulumNet();
            Assert.AreEqual(-Math.Sin(1.0), net.Evaluate(1.0), 0.03);
            Assert.AreEqual(0.0, net.Evaluate(0.0), 0.03);
        }

        [Test]
        public void Periods_MatchNotebook()
        {
            var t = Pendulum.True(); var n = Pendulum.Learned(TrainedWeights.PendulumNet());
            for (int i = 0; i < 1000; i++) { t.Step(); n.Step(); }
            Assert.AreEqual(8.35, t.MeasuredPeriod, 0.03);
            Assert.AreEqual(8.34, n.MeasuredPeriod, 0.03);
        }

        [Test]
        public void Energy_NearlyConserved_OverLongRun()
        {
            var n = Pendulum.Learned(TrainedWeights.PendulumNet());
            double e0 = n.Energy(), maxDev = 0;
            for (int i = 0; i < 4000; i++) { n.Step(); maxDev = Math.Max(maxDev, Math.Abs(n.Energy() - e0)); }
            Assert.Less(maxDev, 0.01);
        }
    }

    public class TinyNetTests
    {
        [Test]
        public void JsonRoundTrip_GivesSameOutput()
        {
            var a = TrainedWeights.CartPoleNet();
            var b = TinyNet.FromJson(a.ToJson("cartpole"));
            var x = new[] { 0.1, -0.2, 0.3, -0.4 };
            Assert.AreEqual(a.Evaluate(x)[0], b.Evaluate(x)[0], 1e-15);
        }
    }
}
