# Cart-pole and learned pendulum — Unity 6

The 3D version of Sections 8 and 9 of the Computational Classnotes notebook
**Feedforward Neural Networks: The Algebra and the Computing**:

1. **Cart-pole.** A 4-8-1 network pushes a cart along a ±3 m rail and keeps a 1 m stick upright.
   Hit the stick and watch it recover, switch to the LQR teacher, or switch the controller off
   and watch the stick fall.
2. **Learned pendulum.** Two pendulums swing from 115°: the true one (gray, a = −sin θ) and one
   driven by a 1-6-1 network that learned the force law (orange). They stay in step.

The physics is our own code (RK4 for the cart-pole, leapfrog for the pendulum; cart-pole
equations from Barto, Sutton and Anderson 1983), in `double`,
not Unity's PhysX, so the motion matches the notebook. Unity only draws the objects.

Both networks can also be **trained inside Unity** (a third, teal option next to the notebook's
weights), with snapshots during training and a replay that shows the early networks dropping the
stick and the later ones balancing it. The notebook's networks stay the default and are never changed.

The scene is a dark studio stage (grid floor, fog, soft shadows) seen through a 3D camera you can
orbit and zoom, within limits.

© Computational Classnotes · CC BY-NC-SA 4.0

---

## Quick start

1. In Unity Hub, create a new project with **Unity 6** and the **Universal 3D** template
   (the built-in-pipeline **3D** template works too).
2. Copy this repository's `Assets/` folder into the project (merge it with the project's own `Assets/`).
3. In the Editor menu choose **Computational Classnotes ▸ Create Cart-Pole & Pendulum Scene**.
   This saves `Assets/CCN-CartPole-Pendulum/Scenes/CartPolePendulum.unity` and adds it to the build.
4. Press **Play**.

Alternative to step 3: add the `CCNDemo` component to an empty GameObject in any scene.

The scene is built at runtime from primitives, so there are no prefabs or materials to set up.
Keyboard input works with either the Input System package or the old Input Manager.

## Controls

| Key | Button | Action |
|---|---|---|
| Space | Play / Pause | run or pause the simulation |
| R | Reset | cart-pole back to (0, 0, 0.3, 0); pendulums back to θ = 2.0 |
| Tab | Cart-pole / Learned pendulum | switch scene |
| ← / → | Push | add −2 / +2 rad/s to ω |
| 1 / 2 / 3 / 4 | Notebook / Unity net / LQR / Off | cart-pole controller |
| 1 / 2 | Notebook net / Unity net | which network drives the pendulum |
| P | Run the notebook's push test | start at θ = 0.05; +2 rad/s at t = 1.5 s, −2 rad/s at t = 4 s |
| O | Same pivot | pendulums from one pivot, or side by side |
| T | Train / Stop training | native training for the current scene |
| V | Reset view | camera back to the front view |
| C | Credits | show the credits screen |

**Camera:** drag with the left or right mouse button to look around, scroll to zoom. The camera
turns at most 80° to either side, stays above the floor, keeps between 2.2 m and 10 m from the demo,
and never pans away from it. Drags and scrolls that start on the control panel are ignored.

Speed: ¼× to 4×. At 1×, one cart-pole step (0.02 s) runs per `FixedUpdate`, and the pendulum
runs at 1 time unit per second (`pendulumUnitsPerSecond` on the component).

The stick's colour shows who is in control: orange for the notebook network, teal for the
Unity-trained network, light blue for LQR, gray for Off.

## Training in Unity

The recipes follow the notebook (`Recipes` in `Runtime/Core/Training.cs`):

| | cart-pole 4-8-1 | pendulum 1-6-1 |
|---|---|---|
| data | 3000 states, x∈[−1,1], v∈[−2,2], θ∈[−0.5,0.5], ω∈[−2,2]; target = LQR / 10 | 200 angles in [−2.6, 2.6]; target −sin θ + noise (sd 0.02) |
| start | weights normal, sd 1/√(fan-in); zero biases | same |
| optimizer | full-batch gradient descent, learning rate 0.5 | Adam (0.9, 0.999, 1e−8), learning rate 0.03 |
| epochs | 4000 | 2000 |
| snapshots | 0, 20, 40, 100, 200, 500, 1000, 2000, 4000 | 0, 25, 50, 100, 200, 500, 1000, 2000 |

The loss is the mean squared error, L = (1/N) Σ (out − y)², with exact backpropagation (checked against
finite differences in the tests). Training runs on a background thread, so the demo keeps running;
the panel shows the epoch, the loss, a progress bar and a live plot of log₁₀ loss. On a desktop
CPU the cart-pole run takes a few seconds and the pendulum run well under a second.

- **Snapshots:** click an epoch to put that snapshot in charge of "Unity net" and restart from θ = 0.3
  (cart-pole) or θ = 2.0 (pendulum).
- **Replay training** (cart-pole): every snapshot gets 6 s from θ = 0.3 in turn, and the panel lists
  which ones stayed up (✓) and which fell (✗). Run outside Unity, this gives
  0 ✗, 20 ✗, 40 ✗, 100 ✓, 200 ✓ … 4000 ✓, the same pattern as the notebook's figures.
- **Save final weights as JSON:** writes `cartpole_net_unity.json` or `pendulum_net_unity.json`
  (in the Editor to `Assets/StreamingAssets/CCN/`, in a build to `Application.persistentDataPath`).
  A saved file is loaded at the next start as "Unity net" (`loadSavedUnityNets` on the component).

The random numbers come from C#'s `System.Random` (seeds 3, 5, 11 and 1, as in the notebook), not
Wolfram's `SeedRandom`, so the starting weights and data differ from the notebook's and the learned
weights are different numbers. They behave the same way: final losses about 0.00004 (cart-pole) and
0.0003 (pendulum), period 8.354 for the Unity-trained pendulum.

## Layout

```
Assets/
  CCN-CartPole-Pendulum/
    Runtime/Core/          plain C# physics, no UnityEngine (assembly CCN.Core)
      CartPole.cs          equations, RK4, fallen check, kick
      Controllers.cs       NetworkController, LqrController, OffController
      Pendulum.cs          leapfrog, energy, period measurement
      TinyNet.cs           sigmoid-hidden / linear-output network, JSON layout
      TrainedWeights.cs    the notebook's trained weights and LQR gains
      Training.cs          Trainer (backprop, gradient descent, Adam) and the notebook's recipes
    Runtime/Views/         MonoBehaviour and drawing
      CCNDemo.cs           builds the scene, FixedUpdate stepping, training UI, credits
      OrbitCamera.cs       limited orbit and zoom camera
      TrainingSession.cs   background training thread, snapshots, save/load
      InputUtil.cs         keyboard and mouse for either input backend
      Shapes.cs            primitives, palette, grid floor, studio lighting
      CartPoleView.cs      cart, rail, stick, force arrow
      PendulumView.cs      frame, two pendulums
      LivePlot.cs          θ(t) plot
      WeightLoader.cs      reads StreamingAssets/CCN/*.json
    Editor/CreateDemoScene.cs   menu items
    Tests/EditMode/        NUnit tests of the sanity checks (assembly CCN.Core.Tests)
  StreamingAssets/CCN/
    cartpole_net.json      4-8-1 weights
    pendulum_net.json      1-6-1 weights
```

## Swapping in retrained weights

`WeightLoader` reads `Assets/StreamingAssets/CCN/cartpole_net.json` and `pendulum_net.json` at
startup and falls back to the built-in weights if a file is missing or has the wrong shape (the
panel says which was used). Each layer stores `W` row-major, with rows = neurons of the next layer
and columns = inputs, and z = W·a + b. `inputScale` multiplies the raw input
(cart-pole: 1, 1/2, 2, 1/2, i.e. x/1, v/2, θ/0.5, ω/2; pendulum: 1/2).
The menu item **Write Built-in Weights to StreamingAssets** regenerates both files.

On Android and WebGL, StreamingAssets isn't a plain folder, so those builds use the built-in weights.

## Checks

Run them in **Window ▸ General ▸ Test Runner ▸ EditMode** (12 tests; the cart-pole training test takes
a few seconds). The same code was compiled and run outside Unity, and gave:

| check | result |
|---|---|
| network from (0, 0, 0.3, 0), 300 steps | stays up, final θ = 0.0006 rad, max \|x\| = 0.69 m |
| u = 0 from (0, 0, 0.3, 0) | falls at t = 0.62 s |
| LQR teacher, 300 steps | stays up, final θ = 0.0008 rad |
| push test (±2 rad/s at steps 75 and 200) | recovers, max \|x\| = 1.38 m |
| pendulum network(1.0), network(0) | −0.8434 (−sin 1 = −0.8415), −0.0004 |
| period from θ = 2.0: true / network | 8.349 / 8.339 |
| energy drift over 200 time units (leapfrog, h = 0.05) | < 0.0007 for both |
| backprop vs finite differences | equal to 7 digits |
| Unity training, cart-pole snapshots from θ = 0.3 | epochs 0, 20, 40 fall; 100 and later balance |
| Unity training, pendulum | net(1) = −0.844, period 8.354 |

## Ideas for later

- A semi-transparent gray pendulum in the same-pivot layout (needs a transparent URP material).

## Credits

- **Dara O Shayda** · Republic of Ireland
- **Noor Hoshino** (نور 星野 · نور هوشينو · ノール・星野) · Advanced AI System Programming · Hokkaido Prefecture, Japan

© Computational Classnotes · CC BY-NC-SA 4.0

## License

© Computational Classnotes · CC BY-NC-SA 4.0. See [LICENSE.md](LICENSE.md).
