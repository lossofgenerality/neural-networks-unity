// Computational Classnotes — Cart-pole and learned pendulum (Unity 6)
// Main component: builds the scene, steps our own physics in FixedUpdate, runs native training,
// drives the orbit camera and draws the UI.
// Add it to an empty GameObject in any scene (or use the menu
// Computational Classnotes ▸ Create Cart-Pole & Pendulum Scene) and press Play.
// © Computational Classnotes · CC BY-NC-SA 4.0

using System;
using System.Collections.Generic;
using System.Text;
using CCN.Core;
using CCN.Views;
using UnityEngine;

[DisallowMultipleComponent]
public sealed class CCNDemo : MonoBehaviour
{
    public enum Mode { CartPole, Pendulum }
    public enum Controller { NotebookNet, UnityNet, LQR, Off }
    public enum PendulumSource { NotebookNet, UnityNet }

    [Header("Start")]
    public Mode startMode = Mode.CartPole;
    public Controller startController = Controller.NotebookNet;
    [Tooltip("Initial tilt for Reset (rad). Notebook: 0.3 (≈17°).")]
    public double startTheta = 0.3;
    [Tooltip("Initial pendulum angle (rad). Notebook: 2.0 (≈115°).")]
    public double pendulumTheta0 = 2.0;
    [Tooltip("Δω added by a push (rad/s).")]
    public double pushStrength = 2.0;
    [Tooltip("Pendulum time units per second of screen time at 1× speed.")]
    public double pendulumUnitsPerSecond = 1.0;
    [Tooltip("Load a previously saved Unity-trained network at start, if one exists.")]
    public bool loadSavedUnityNets = true;

    // --- simulation ---
    Mode _mode;
    Controller _ctrl;
    PendulumSource _penSource = PendulumSource.NotebookNet;
    CartPole _cp;
    ICartPoleController _notebookCtrl, _unityCtrl, _lqr, _off;
    Pendulum _pTrue, _pNet;
    TinyNet _notebookPenNet, _unityPenNet;
    string _cartNetSource, _penNetSource;
    bool _playing = true;
    float _speed = 1f;
    double _accum;
    bool _pushTest, _samePivot, _showForce = true, _showCredits;

    // --- native training ---
    TrainingSession _cpTrain, _penTrain;
    int _cpSnap = -1, _penSnap = -1;
    bool _cpWasRunning, _penWasRunning;
    LivePlot _cpLossPlot, _penLossPlot;
    readonly List<Vector2> _lossBuf = new List<Vector2>();
    string _saveMsg = "";
    // training replay (cart-pole): each snapshot tries to balance from the start state in turn
    bool _replay;
    int _replayIndex, _replayHold;
    readonly StringBuilder _replayLog = new StringBuilder();

    // --- views ---
    CartPoleView _cpView;
    PendulumView _penView;
    LivePlot _cpPlot, _penPlot;
    Camera _cam;
    OrbitCamera _orbit;
    Vector2 _scroll;

    static readonly float[] Speeds = { 0.25f, 0.5f, 1f, 2f, 4f };
    static readonly string[] SpeedLabels = { "¼×", "½×", "1×", "2×", "4×" };
    const float PanelWidth = 350f;
    const int PlotW = 310, PlotH = 100;
    public const string Copyright = "© Computational Classnotes · CC BY-NC-SA 4.0";

    static readonly Color LqrColor = new Color(0.55f, 0.70f, 1f);
    static readonly Color OffColor = new Color(0.55f, 0.56f, 0.60f);

    // ------------------------------------------------------------------ setup

    void Awake()
    {
        Time.fixedDeltaTime = (float)CartPole.Dt;   // 0.02 s; one RK4 step per FixedUpdate at 1×

        var cartNet = WeightLoader.Load("cartpole_net.json", TrainedWeights.CartPoleNet, 4, 1, out _cartNetSource);
        _notebookPenNet = WeightLoader.Load("pendulum_net.json", TrainedWeights.PendulumNet, 1, 1, out _penNetSource);
        _notebookCtrl = new NetworkController(cartNet);
        _lqr = new LqrController();
        _off = new OffController();

        _cpTrain = new TrainingSession("cart-pole 4-8-1", "cartpole_net_unity.json",
            () => Recipes.CartPoleStart(5), () => Recipes.CartPoleData(3000, 3), Recipes.CartPoleConfig);
        _penTrain = new TrainingSession("pendulum 1-6-1", "pendulum_net_unity.json",
            () => Recipes.PendulumStart(1), () => Recipes.PendulumData(200, 11), Recipes.PendulumConfig);
        if (loadSavedUnityNets)
        {
            if (_cpTrain.TryLoad(4, 1)) SelectCartSnapshot(0, false);
            if (_penTrain.TryLoad(1, 1)) SelectPendulumSnapshot(0, false);
        }

        _cp = new CartPole(new CartPoleState(0, 0, startTheta, 0));
        _pTrue = Pendulum.True(pendulumTheta0);
        _pNet = Pendulum.Learned(_notebookPenNet, pendulumTheta0);

        _cpView = new CartPoleView(transform);
        _penView = new PendulumView(transform);
        _cpPlot = new LivePlot(PlotW, PlotH, 300, -Mathf.PI / 2, Mathf.PI / 2, Palette.Orange);           // 6 s window
        _penPlot = new LivePlot(PlotW, PlotH, 400, -Mathf.PI, Mathf.PI, Palette.Gray, Palette.Orange);    // 20 time units
        _cpLossPlot = NewLossPlot(_cpTrain.Config, Palette.Teal);
        _penLossPlot = NewLossPlot(_penTrain.Config, Palette.Teal);

        SetupCamera();
        _ctrl = startController == Controller.UnityNet && _unityCtrl == null ? Controller.NotebookNet : startController;
        SetMode(startMode, true);
        ResetAll();
    }

    static LivePlot NewLossPlot(TrainerConfig c, Color color) =>
        new LivePlot(PlotW, 80, c.Epochs / c.LogEvery + 1, -5f, 0.5f, color);   // log10(loss)

    void SetupCamera()
    {
        _cam = Camera.main;
        if (_cam == null)
        {
            var go = new GameObject("Main Camera") { tag = "MainCamera" };
            _cam = go.AddComponent<Camera>();
            go.AddComponent<AudioListener>();
        }
        _cam.fieldOfView = 38f;
        _cam.nearClipPlane = 0.05f;
        _cam.farClipPlane = 60f;
        Shapes.StudioLighting(_cam);
        _orbit = new OrbitCamera(_cam);
    }

    void SetMode(Mode m, bool snapCamera = false)
    {
        _mode = m;
        _cpView.SetActive(m == Mode.CartPole);
        _penView.SetActive(m == Mode.Pendulum);
        if (m == Mode.CartPole)
            _orbit.Frame(new Vector3(0, 0.55f, 0), 0, 8, 6.6f, 2.2f, 10f, 80f, 60f, snapCamera);
        else
            _orbit.Frame(new Vector3(0, 0.95f, 0), 0, 4, 6.4f, 2.2f, 10f, 80f, 60f, snapCamera);
        _accum = 0;
        RefreshColors();
    }

    void ResetCartPole(double theta)
    {
        _cp.Reset(new CartPoleState(0, 0, theta, 0));
        _cpPlot.Clear(); _cpPlot.Add((float)theta);
        _accum = 0;
    }

    void ResetPendulums()
    {
        _pTrue.Reset(pendulumTheta0);
        var net = _penSource == PendulumSource.UnityNet && _unityPenNet != null ? _unityPenNet : _notebookPenNet;
        _pNet = Pendulum.Learned(net, pendulumTheta0);
        _penPlot.Clear(); _penPlot.Add((float)_pTrue.Theta, (float)_pNet.Theta);
        _accum = 0;
    }

    void ResetAll()
    {
        _pushTest = false;
        _replay = false;
        ResetCartPole(startTheta);
        ResetPendulums();
    }

    /// <summary>The notebook's push test: start at θ = 0.05, +2 rad/s at step 75 (t = 1.5 s), −2 rad/s at step 200 (t = 4 s).</summary>
    void StartPushTest()
    {
        _replay = false;
        ResetCartPole(0.05);
        _pushTest = true;
        _playing = true;
    }

    ICartPoleController Active
    {
        get
        {
            switch (_ctrl)
            {
                case Controller.NotebookNet: return _notebookCtrl;
                case Controller.UnityNet: return _unityCtrl ?? _off;
                case Controller.LQR: return _lqr;
                default: return _off;
            }
        }
    }

    void SetController(Controller c)
    {
        if (c == Controller.UnityNet && _unityCtrl == null) { _saveMsg = "Train the network in Unity first (Train below, or T)."; return; }
        _ctrl = c;
        _replay = false;
        RefreshColors();
    }

    void RefreshColors()
    {
        Color stick;
        switch (_ctrl)
        {
            case Controller.NotebookNet: stick = Palette.Orange; break;
            case Controller.UnityNet: stick = Palette.Teal; break;
            case Controller.LQR: stick = LqrColor; break;
            default: stick = OffColor; break;
        }
        _cpView.SetStickColor(stick);
        _penView.NetRig.SetColor(_penSource == PendulumSource.UnityNet ? Palette.Teal : Palette.Orange);
    }

    // ------------------------------------------------------------------ training

    void SelectCartSnapshot(int i, bool resetSim)
    {
        var net = _cpTrain.Snapshot(i);
        if (net == null) return;
        _cpSnap = i;
        _unityCtrl = new NetworkController(net);
        if (resetSim) { _ctrl = Controller.UnityNet; _pushTest = false; ResetCartPole(startTheta); RefreshColors(); }
    }

    void SelectPendulumSnapshot(int i, bool resetSim)
    {
        var net = _penTrain.Snapshot(i);
        if (net == null) return;
        _penSnap = i;
        _unityPenNet = net;
        if (resetSim) { _penSource = PendulumSource.UnityNet; ResetPendulums(); RefreshColors(); }
    }

    void StartTraining(TrainingSession s)
    {
        if (s.Running) { s.Cancel(); return; }
        _saveMsg = "";
        if (s == _cpTrain) { _replay = false; _cpLossPlot = NewLossPlot(s.Config, Palette.Teal); _cpSnap = -1; }
        else { _penLossPlot = NewLossPlot(s.Config, Palette.Teal); _penSnap = -1; }
        s.Start();
    }

    void PollTraining()
    {
        Drain(_cpTrain, _cpLossPlot);
        Drain(_penTrain, _penLossPlot);

        // When a run finishes, put its final network in charge of the Unity option.
        if (_cpWasRunning && !_cpTrain.Running && _cpTrain.SnapshotCount > 0)
        {
            SelectCartSnapshot(_cpTrain.SnapshotCount - 1, false);
            if (_ctrl == Controller.UnityNet) RefreshColors();
        }
        if (_penWasRunning && !_penTrain.Running && _penTrain.SnapshotCount > 0)
            SelectPendulumSnapshot(_penTrain.SnapshotCount - 1, _penSource == PendulumSource.UnityNet);
        _cpWasRunning = _cpTrain.Running;
        _penWasRunning = _penTrain.Running;
    }

    void Drain(TrainingSession s, LivePlot plot)
    {
        _lossBuf.Clear();
        s.DrainLoss(_lossBuf);
        foreach (var p in _lossBuf) plot.Add(Mathf.Log10(Mathf.Max(p.y, 1e-6f)));
    }

    void StartReplay()
    {
        if (_cpTrain.SnapshotCount == 0) { _saveMsg = "Train first: the replay needs the snapshots."; return; }
        _replay = true; _replayIndex = 0; _replayHold = 0; _replayLog.Length = 0;
        _playing = true;
        SelectCartSnapshot(0, true);
    }

    void AdvanceReplay()
    {
        int epoch = _cpTrain.SnapshotEpochs[_replayIndex];
        _replayLog.Append(_replayLog.Length == 0 ? "" : "  ").Append(epoch).Append(_cp.Fallen ? " ✗" : " ✓");
        _replayIndex++; _replayHold = 0;
        if (_replayIndex >= _cpTrain.SnapshotCount) { _replay = false; return; }
        SelectCartSnapshot(_replayIndex, true);
    }

    // ------------------------------------------------------------------ physics

    void FixedUpdate()
    {
        if (!_playing) return;
        // Simulated time added per FixedUpdate is fixed, so the motion is frame-rate independent.
        if (_mode == Mode.CartPole)
        {
            _accum += Time.fixedDeltaTime * _speed;
            while (_accum >= CartPole.Dt - 1e-9) { StepCartPole(); _accum -= CartPole.Dt; }
        }
        else
        {
            _accum += Time.fixedDeltaTime * _speed * pendulumUnitsPerSecond;
            while (_accum >= Pendulum.DefaultH - 1e-9) { StepPendulums(); _accum -= Pendulum.DefaultH; }
        }
    }

    void StepCartPole()
    {
        if (_replay)
        {
            // Each snapshot gets 6 s; a fallen stick stays down for 0.8 s before the next one.
            if (_cp.Fallen) { if (++_replayHold >= 40) AdvanceReplay(); return; }
            if (_cp.StepCount >= 300) { AdvanceReplay(); return; }
        }
        if (_cp.Fallen) return;
        if (_pushTest)
        {
            if (_cp.StepCount == 75) _cp.Kick(+pushStrength);
            if (_cp.StepCount == 200) _cp.Kick(-pushStrength);
        }
        _cp.Step(Active);
        _cpPlot.Add((float)_cp.State.Theta);
    }

    void StepPendulums()
    {
        _pTrue.Step(); _pNet.Step();
        _penPlot.Add((float)_pTrue.Theta, (float)_pNet.Theta);
    }

    // ------------------------------------------------------------------ input and drawing

    bool PointerOverUi
    {
        get
        {
            if (_showCredits) return true;
            return MouseInput.Position.x < PanelWidth * UiScale;
        }
    }

    void Update()
    {
        PollTraining();

        if (KeyInput.Pressed(KeyInput.K.Space)) _playing = !_playing;
        if (KeyInput.Pressed(KeyInput.K.R)) ResetAll();
        if (KeyInput.Pressed(KeyInput.K.Tab)) SetMode(_mode == Mode.CartPole ? Mode.Pendulum : Mode.CartPole);
        if (KeyInput.Pressed(KeyInput.K.C)) _showCredits = !_showCredits;
        if (KeyInput.Pressed(KeyInput.K.V)) _orbit.ResetView();
        if (_mode == Mode.CartPole)
        {
            if (KeyInput.Pressed(KeyInput.K.Left)) _cp.Kick(-pushStrength);
            if (KeyInput.Pressed(KeyInput.K.Right)) _cp.Kick(+pushStrength);
            if (KeyInput.Pressed(KeyInput.K.D1)) SetController(Controller.NotebookNet);
            if (KeyInput.Pressed(KeyInput.K.D2)) SetController(Controller.UnityNet);
            if (KeyInput.Pressed(KeyInput.K.D3)) SetController(Controller.LQR);
            if (KeyInput.Pressed(KeyInput.K.D4)) SetController(Controller.Off);
            if (KeyInput.Pressed(KeyInput.K.P)) StartPushTest();
            if (KeyInput.Pressed(KeyInput.K.T)) StartTraining(_cpTrain);
        }
        else
        {
            if (KeyInput.Pressed(KeyInput.K.O)) { _samePivot = !_samePivot; _penView.SetLayout(_samePivot); }
            if (KeyInput.Pressed(KeyInput.K.D1)) SetPendulumSource(PendulumSource.NotebookNet);
            if (KeyInput.Pressed(KeyInput.K.D2)) SetPendulumSource(PendulumSource.UnityNet);
            if (KeyInput.Pressed(KeyInput.K.T)) StartTraining(_penTrain);
        }
    }

    void SetPendulumSource(PendulumSource s)
    {
        if (s == _penSource) return;
        if (s == PendulumSource.UnityNet && _unityPenNet == null) { _saveMsg = "Train the network in Unity first (Train below, or T)."; return; }
        _penSource = s;
        ResetPendulums();
        RefreshColors();
    }

    void LateUpdate()
    {
        float frac = Mathf.Clamp01(PanelWidth * UiScale / Mathf.Max(1, Screen.width));
        _cam.rect = new Rect(frac, 0, 1 - frac, 1);   // 3D view to the right of the panel
        _orbit.Tick(Time.unscaledDeltaTime, PointerOverUi);

        if (_mode == Mode.CartPole) _cpView.Show(_cp, _showForce);
        else _penView.Show(_pTrue.Theta, _pNet.Theta);
    }

    void OnDestroy()
    {
        _cpTrain?.Shutdown();
        _penTrain?.Shutdown();
    }

    static float UiScale => Mathf.Clamp(Screen.height / 900f, 1f, 2.5f);

    // ------------------------------------------------------------------ UI

    GUIStyle _h1, _h2, _mono, _small, _credits;
    static Texture2D _panelTex;

    void Styles()
    {
        if (_h1 != null) return;
        _h1 = new GUIStyle(GUI.skin.label) { fontSize = 15, fontStyle = FontStyle.Bold, wordWrap = true };
        _h1.normal.textColor = new Color(0.95f, 0.95f, 0.97f);
        _h2 = new GUIStyle(GUI.skin.label) { fontSize = 13, fontStyle = FontStyle.Bold };
        _h2.normal.textColor = Palette.Orange;
        _mono = new GUIStyle(GUI.skin.label) { fontSize = 13, richText = true, wordWrap = true };
        _mono.normal.textColor = new Color(0.86f, 0.88f, 0.92f);
        _small = new GUIStyle(GUI.skin.label) { fontSize = 11, wordWrap = true, richText = true };
        _small.normal.textColor = new Color(0.60f, 0.63f, 0.70f);
        _credits = new GUIStyle(_mono) { wordWrap = true };
        _credits.normal.textColor = Color.white;
        _panelTex = new Texture2D(1, 1); _panelTex.SetPixel(0, 0, new Color(0.105f, 0.115f, 0.145f)); _panelTex.Apply();
    }

    void OnGUI()
    {
        Styles();
        float s = UiScale;
        GUI.matrix = Matrix4x4.Scale(new Vector3(s, s, 1));
        float H = Screen.height / s;

        GUI.DrawTexture(new Rect(0, 0, PanelWidth, H), _panelTex);
        GUILayout.BeginArea(new Rect(10, 8, PanelWidth - 16, H - 16));
        _scroll = GUILayout.BeginScrollView(_scroll, false, false);

        GUILayout.Label(_mode == Mode.CartPole ? "Cart-pole — a 4-8-1 network keeps the stick up" : "Learned pendulum — a 1-6-1 network learned a(θ) ≈ −sin θ", _h1);
        int m = GUILayout.Toolbar((int)_mode, new[] { "Cart-pole", "Learned pendulum" });
        if (m != (int)_mode) SetMode((Mode)m);
        GUILayout.Space(4);

        GUILayout.BeginHorizontal();
        if (GUILayout.Button(_playing ? "Pause (Space)" : "Play (Space)")) _playing = !_playing;
        if (GUILayout.Button("Reset (R)")) ResetAll();
        GUILayout.EndHorizontal();
        GUILayout.BeginHorizontal();
        GUILayout.Label("Speed", GUILayout.Width(44));
        int si = Array.IndexOf(Speeds, _speed);
        _speed = Speeds[GUILayout.Toolbar(si < 0 ? 2 : si, SpeedLabels)];
        GUILayout.EndHorizontal();
        GUILayout.Space(6);

        if (_mode == Mode.CartPole) CartPolePanel(); else PendulumPanel();

        GUILayout.Space(8);
        GUILayout.Label("View", _h2);
        GUILayout.BeginHorizontal();
        GUILayout.Label("Drag to look around, scroll to zoom.", _small);
        if (GUILayout.Button("Reset view (V)", GUILayout.Width(110))) _orbit.ResetView();
        GUILayout.EndHorizontal();
        GUILayout.Space(6);
        if (GUILayout.Button(_showCredits ? "Hide credits (C)" : "Credits (C)")) _showCredits = !_showCredits;
        GUILayout.Label(Copyright, _small);

        GUILayout.EndScrollView();
        GUILayout.EndArea();

        if (_showCredits) DrawCredits(Screen.width / s, H);
    }

    void CartPolePanel()
    {
        GUILayout.Label("Controller (1–4)", _h2);
        int c = GUILayout.Toolbar((int)_ctrl, new[] { "Notebook", "Unity net", "LQR", "Off" });
        if (c != (int)_ctrl) SetController((Controller)c);
        GUILayout.BeginHorizontal();
        if (GUILayout.Button("◀ Push (←)")) _cp.Kick(-pushStrength);
        if (GUILayout.Button("Push (→) ▶")) _cp.Kick(+pushStrength);
        GUILayout.EndHorizontal();
        if (GUILayout.Button(_pushTest ? "Push test running… (P restarts)" : "Run the notebook's push test (P)")) StartPushTest();
        _showForce = GUILayout.Toggle(_showForce, " Show force arrow");
        GUILayout.Space(4);

        var st = _cp.State;
        string who = Active.Name;
        if (_ctrl == Controller.UnityNet && _cpSnap >= 0) who = "Unity net, epoch " + _cpTrain.SnapshotEpochs[_cpSnap];
        else if (_ctrl == Controller.NotebookNet) who = "Notebook net";
        GUILayout.Label(
            $"t = {_cp.Time:F2} s    u = {_cp.LastForce:F3} N\n" +
            $"θ = {st.Theta:F4} rad ({st.Theta * 180 / Math.PI:F1}°)    ω = {st.Omega:F4} rad/s\n" +
            $"x = {st.X:F4} m    v = {st.V:F4} m/s\n" +
            $"in control: {who}", _mono);
        if (_cp.Fallen) GUILayout.Label("<color=#ff6b5e><b>The stick has fallen (|θ| > π/2). Press Reset.</b></color>", _mono);
        if (Mathf.Abs((float)st.X) > 3f) GUILayout.Label("<color=#ff6b5e>The cart has left the ±3 m rail.</color>", _mono);

        GUILayout.Label("θ(t), last 6 s  (range ±π/2)", _small);
        _cpPlot.Refresh();
        GUILayout.Box(_cpPlot.Texture, GUILayout.Width(PlotW), GUILayout.Height(PlotH));
        GUILayout.Label($"Notebook weights: {_cartNetSource}", _small);

        GUILayout.Space(8);
        GUILayout.Label("Train in Unity (T)", _h2);
        GUILayout.Label("3000 random states, target = LQR teacher / 10, full-batch gradient descent, learning rate 0.5, 4000 epochs. " +
                        "Runs in the background; the notebook's network is not changed.", _small);
        TrainingBlock(_cpTrain, _cpLossPlot);

        int n = _cpTrain.SnapshotCount;
        if (n > 0)
        {
            GUILayout.Label("Snapshot in charge of “Unity net” (click to try it from θ = 0.3):", _small);
            int pick = GUILayout.SelectionGrid(_cpSnap, EpochLabels(_cpTrain, _cpTrain.FromFile), 5);
            if (pick != _cpSnap) { _replay = false; SelectCartSnapshot(pick, true); }
            if (!_cpTrain.FromFile && !_cpTrain.Running)
            {
                if (GUILayout.Button(_replay ? $"Replaying… epoch {_cpTrain.SnapshotEpochs[_replayIndex]}" : "Replay training: every snapshot tries in turn"))
                    StartReplay();
                if (_replayLog.Length > 0) GUILayout.Label("Stayed up (✓) or fell (✗) by epoch: " + _replayLog, _small);
            }
            SaveButton(_cpTrain);
        }
    }

    void PendulumPanel()
    {
        bool sp = GUILayout.Toggle(_samePivot, " Same pivot (O), otherwise side by side");
        if (sp != _samePivot) { _samePivot = sp; _penView.SetLayout(sp); }
        GUILayout.Label("Network pendulum driven by (1–2)", _h2);
        int ps = GUILayout.Toolbar((int)_penSource, new[] { "Notebook net", "Unity net" });
        if (ps != (int)_penSource) SetPendulumSource((PendulumSource)ps);
        GUILayout.Space(4);

        string netName = _penSource == PendulumSource.UnityNet && _penSnap >= 0
            ? $"<color=#33bdb3><b>Unity net, epoch {_penTrain.SnapshotEpochs[_penSnap]}</b></color>"
            : "<color=#f5852a><b>Notebook net</b></color>";
        GUILayout.Label(
            $"t = {_pTrue.Time:F2}   (h = {Pendulum.DefaultH})\n" +
            $"<color=#b8bac0><b>True</b></color>, a = −sin θ\n" +
            $"   θ = {_pTrue.Theta:F4}   ω = {_pTrue.Omega:F4}\n" +
            $"   E = {_pTrue.Energy():F5}   T = {P(_pTrue.MeasuredPeriod)}\n" +
            $"{netName}, a = net(θ)\n" +
            $"   θ = {_pNet.Theta:F4}   ω = {_pNet.Omega:F4}\n" +
            $"   E = {_pNet.Energy():F5}   T = {P(_pNet.MeasuredPeriod)}\n" +
            $"Δθ = {_pNet.Theta - _pTrue.Theta:F4} rad", _mono);
        GUILayout.Label("Expected periods: true ≈ 8.35, notebook net ≈ 8.34 (small-angle law: 6.28). " +
                        "E is conserved because the learned acceleration depends on θ only.", _small);
        GUILayout.Label("θ(t), last 20 time units  (gray true, coloured network)", _small);
        _penPlot.Refresh();
        GUILayout.Box(_penPlot.Texture, GUILayout.Width(PlotW), GUILayout.Height(PlotH));
        GUILayout.Label($"Notebook weights: {_penNetSource}", _small);

        GUILayout.Space(8);
        GUILayout.Label("Train in Unity (T)", _h2);
        GUILayout.Label("200 angles in [−2.6, 2.6], targets −sin θ plus noise (sd 0.02), Adam with learning rate 0.03, 2000 epochs. " +
                        "The notebook's network is not changed.", _small);
        TrainingBlock(_penTrain, _penLossPlot);
        if (_penTrain.SnapshotCount > 0)
        {
            GUILayout.Label("Snapshot in charge of “Unity net” (click to swing it):", _small);
            int pick = GUILayout.SelectionGrid(_penSnap, EpochLabels(_penTrain, _penTrain.FromFile), 4);
            if (pick != _penSnap) SelectPendulumSnapshot(pick, true);
            SaveButton(_penTrain);
        }
    }

    void TrainingBlock(TrainingSession s, LivePlot lossPlot)
    {
        GUILayout.BeginHorizontal();
        if (GUILayout.Button(s.Running ? "Stop training" : (s.HasResult && !s.FromFile ? "Train again" : "Train"), GUILayout.Width(110))) StartTraining(s);
        GUILayout.Label(s.Running ? $"epoch {s.Epoch} / {s.Config.Epochs}   loss {s.LastLoss:F6}" : s.Status, _small);
        GUILayout.EndHorizontal();

        // progress bar
        var r = GUILayoutUtility.GetRect(PlotW, 6, GUILayout.Width(PlotW));
        var old = GUI.color;
        GUI.color = new Color(0.2f, 0.22f, 0.27f); GUI.DrawTexture(r, Texture2D.whiteTexture);
        float f = s.Config.Epochs > 0 ? Mathf.Clamp01(s.Epoch / (float)s.Config.Epochs) : 0f;
        if (!s.Running && s.HasResult) f = 1f;
        GUI.color = Palette.Teal; GUI.DrawTexture(new Rect(r.x, r.y, r.width * f, r.height), Texture2D.whiteTexture);
        GUI.color = old;

        if (s.Running || (s.HasResult && !s.FromFile))
        {
            GUILayout.Label("log₁₀ loss by epoch  (−5 to 0.5)", _small);
            lossPlot.Refresh();
            GUILayout.Box(lossPlot.Texture, GUILayout.Width(PlotW), GUILayout.Height(80));
        }
    }

    void SaveButton(TrainingSession s)
    {
        if (s.Running || s.FromFile) return;
        if (GUILayout.Button("Save final weights as JSON")) s.Save(out _saveMsg);
        if (!string.IsNullOrEmpty(_saveMsg)) GUILayout.Label(_saveMsg, _small);
    }

    static string[] EpochLabels(TrainingSession s, bool fromFile)
    {
        var e = s.SnapshotEpochs;
        var l = new string[e.Length];
        for (int i = 0; i < e.Length; i++) l[i] = fromFile ? "saved" : e[i].ToString();
        return l;
    }

    static string P(double p) { return double.IsNaN(p) ? "…" : p.ToString("F3"); }

    void DrawCredits(float W, float H)
    {
        var r = new Rect((W - 400) / 2 + PanelWidth / 2, (H - 300) / 2, 400, 300);
        GUI.Box(r, GUIContent.none);
        GUI.Box(r, GUIContent.none);
        GUILayout.BeginArea(new Rect(r.x + 18, r.y + 14, r.width - 36, r.height - 28));
        GUILayout.Label("Credits", _h1);
        GUILayout.Label(
            "Dara O Shayda\n" +
            "Republic of Ireland\n\n" +
            "Noor Hoshino\n" +
            "AI System Programming Group\n" +
            "Hokkaido Prefecture, Japan\n\n" +
            "<b>" + Copyright + "</b>", _credits);
        GUILayout.FlexibleSpace();
        if (GUILayout.Button("Close")) _showCredits = false;
        GUILayout.EndArea();
    }
}
