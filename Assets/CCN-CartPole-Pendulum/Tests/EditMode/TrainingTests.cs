// Computational Classnotes — Cart-pole and learned pendulum (Unity 6)
// Edit-mode tests for native training: gradient check, and the notebook's training outcomes.
// © Computational Classnotes · CC BY-NC-SA 4.0

using System;
using NUnit.Framework;

namespace CCN.Core.Tests
{
    public class TrainingTests
    {
        static bool Balances(TinyNet net)
        {
            var cp = new CartPole(new CartPoleState(0, 0, 0.3, 0));
            var c = new NetworkController(net);
            for (int i = 0; i < 300; i++) cp.Step(c);
            return !cp.Fallen && Math.Abs(cp.State.Theta) < 0.1;
        }

        [Test]
        public void Backprop_MatchesFiniteDifferences()
        {
            var tr = new Trainer(Recipes.CartPoleStart(5), Recipes.CartPoleData(50, 3), new TrainerConfig { LearningRate = 1e-7, Epochs = 1 });
            double w0 = tr.Net.W[0][2, 1], h = 1e-6;
            tr.Net.W[0][2, 1] = w0 + h; double lp = tr.Loss();
            tr.Net.W[0][2, 1] = w0 - h; double lm = tr.Loss();
            tr.Net.W[0][2, 1] = w0;
            double numeric = (lp - lm) / (2 * h);
            tr.Step();
            double analytic = (w0 - tr.Net.W[0][2, 1]) / 1e-7;
            Assert.AreEqual(numeric, analytic, 1e-6 * Math.Max(1, Math.Abs(numeric)));
        }

        [Test]
        public void CartPoleTraining_DropsEarly_BalancesFromEpoch100()   // takes a few seconds
        {
            var cfg = Recipes.CartPoleConfig();
            cfg.Epochs = 500; cfg.SnapshotEpochs = new[] { 0, 40, 100, 500 };
            var tr = new Trainer(Recipes.CartPoleStart(5), Recipes.CartPoleData(3000, 3), cfg);
            var results = new System.Collections.Generic.Dictionary<int, bool>();
            tr.Run(null, (e, net) => results[e] = Balances(net), null);
            Assert.IsFalse(results[0], "the untrained network should drop the stick");
            Assert.IsFalse(results[40], "epoch 40 should still drop the stick");
            Assert.IsTrue(results[100], "epoch 100 should balance");
            Assert.IsTrue(results[500], "epoch 500 should balance");
        }

        [Test]
        public void PendulumTraining_LearnsMinusSin()
        {
            var tr = new Trainer(Recipes.PendulumStart(1), Recipes.PendulumData(200, 11), Recipes.PendulumConfig());
            tr.Run(null, null, null);
            Assert.Less(tr.Loss(), 0.001);
            Assert.AreEqual(-Math.Sin(1.0), tr.Net.Evaluate(1.0), 0.03);
            var p = Pendulum.Learned(tr.Net);
            for (int i = 0; i < 1000; i++) p.Step();
            Assert.AreEqual(8.35, p.MeasuredPeriod, 0.05);
        }
    }
}
