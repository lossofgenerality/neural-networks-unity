// Computational Classnotes — Cart-pole and learned pendulum (Unity 6)
// Trained weights from the notebook (Sections 8–9). Used when no JSON file is found.
// © Computational Classnotes · CC BY-NC-SA 4.0

namespace CCN.Core
{
    public static class TrainedWeights
    {
        /// <summary>
        /// Cart-pole controller, 4-8-1. Input (x, v, θ, ω) scaled by (1/1, 1/2, 1/0.5, 1/2).
        /// The raw output is multiplied by 10 and clamped to ±10 N by <see cref="NetworkController"/>.
        /// Trained on 3000 states against LQR/10, gradient descent lr 0.5, 4000 epochs; loss ≈ 0.00097.
        /// </summary>
        public static TinyNet CartPoleNet() => new TinyNet(
            new[]
            {
                new double[,]
                {
                    {  1.7125390,  0.6802361,  0.1710183,  0.2228485 },
                    {  0.4195820, -0.8639005,  1.0074240, -0.1802207 },
                    { -0.5116100, -1.3548090, -3.0196727, -3.0215438 },
                    {  0.1062752,  1.0523612, -0.2151324,  0.0279761 },
                    {  0.7107849,  0.2786495,  1.2690457,  0.6424504 },
                    {  0.4348188,  1.6763395,  3.5779060,  3.6929801 },
                    {  0.4842372, -0.6407557,  0.0220804, -1.0612345 },
                    { -0.0291893, -0.2285702,  0.9444850,  0.6817494 },
                },
                new double[,]
                {
                    { 0.0999654, 0.0963410, -1.0430366, -0.0088368, -0.9592313, 1.8223406, 0.4696063, -0.2815391 },
                },
            },
            new[]
            {
                new[] { 0.0638574, -0.0158166, -0.0103315, -0.0244673, 0.0125493, 0.0012278, -0.0328335, 0.0237862 },
                new[] { -0.0915542 },
            },
            new[] { 1.0 / 1.0, 1.0 / 2.0, 1.0 / 0.5, 1.0 / 2.0 });

        /// <summary>
        /// Learned pendulum force law, 1-6-1, a(θ) ≈ −sin θ. Input θ scaled by 1/2.
        /// Trained on 200 noisy samples in [−2.6, 2.6] with Adam, 2000 epochs; loss ≈ 0.00021.
        /// </summary>
        public static TinyNet PendulumNet() => new TinyNet(
            new[]
            {
                new double[,] { { 3.6462451 }, { 1.0997180 }, { 3.3249025 }, { 2.9595168 }, { 2.7921721 }, { 3.0706169 } },
                new double[,] { { 1.8398602, 1.3670851, 1.9239168, -0.8204751, -2.1562992, -1.1687020 } },
            },
            new[]
            {
                new[] { 4.4307899, -0.2202203, -4.3653518, -0.7225631, 0.8905902, -0.9248930 },
                new[] { -0.3224533 },
            },
            new[] { 0.5 });

        /// <summary>LQR gains from LQRegulatorGains with Q = diag(1,1,10,1), R = 0.1.</summary>
        public static readonly double[] LqrK = { -3.1622777, -5.9144363, -49.538272, -12.846079 };
    }
}
