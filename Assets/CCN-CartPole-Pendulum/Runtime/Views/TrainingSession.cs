// Computational Classnotes — Cart-pole and learned pendulum (Unity 6)
// Runs a native training job on a worker thread, keeps snapshots and the loss history,
// and saves/loads the result as JSON. The notebook's weights are never touched.
// © Computational Classnotes · CC BY-NC-SA 4.0

using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using CCN.Core;
using UnityEngine;

namespace CCN.Views
{
    public sealed class TrainingSession
    {
        public readonly string Title, FileName;
        readonly Func<TinyNet> _start;
        readonly Func<TrainingData> _data;
        readonly Func<TrainerConfig> _config;

        readonly object _lock = new object();
        readonly List<KeyValuePair<int, TinyNet>> _snapshots = new List<KeyValuePair<int, TinyNet>>();
        readonly List<Vector2> _pendingLoss = new List<Vector2>();   // (epoch, loss) not yet drained by the UI
        Thread _thread;
        volatile bool _cancel;

        public TrainerConfig Config { get; private set; }
        public volatile bool Running;
        public volatile int Epoch;
        public double LastLoss = double.NaN;
        public string Status = "Not trained yet.";
        public float Seconds;
        /// <summary>Increments whenever the snapshot list changes, so the UI can refresh.</summary>
        public volatile int Version;
        /// <summary>True when the current result was loaded from a saved file rather than trained in this session.</summary>
        public bool FromFile { get; private set; }

        public TrainingSession(string title, string fileName, Func<TinyNet> start, Func<TrainingData> data, Func<TrainerConfig> config)
        {
            Title = title; FileName = fileName; _start = start; _data = data; _config = config;
            Config = config();
        }

        public bool HasResult { get { lock (_lock) return _snapshots.Count > 0; } }

        public int[] SnapshotEpochs
        {
            get { lock (_lock) { var e = new int[_snapshots.Count]; for (int i = 0; i < e.Length; i++) e[i] = _snapshots[i].Key; return e; } }
        }

        public TinyNet Snapshot(int index)
        {
            lock (_lock) return index >= 0 && index < _snapshots.Count ? _snapshots[index].Value : null;
        }

        public int SnapshotCount { get { lock (_lock) return _snapshots.Count; } }

        /// <summary>Moves the loss points recorded since the last call into 'into'.</summary>
        public void DrainLoss(List<Vector2> into)
        {
            lock (_lock) { into.AddRange(_pendingLoss); _pendingLoss.Clear(); }
        }

        public void Start()
        {
            if (Running) return;
            Config = _config();
            lock (_lock) { _snapshots.Clear(); _pendingLoss.Clear(); }
            FromFile = false;
            _cancel = false; Running = true; Epoch = 0; LastLoss = double.NaN; Seconds = 0;
            Status = "Training…";
            Version++;
            var cfg = Config;
            _thread = new Thread(() =>
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                try
                {
                    var trainer = new Trainer(_start(), _data(), cfg);
                    trainer.Run(
                        (e, l) => { Epoch = e; LastLoss = l; lock (_lock) _pendingLoss.Add(new Vector2(e, (float)l)); Seconds = (float)sw.Elapsed.TotalSeconds; },
                        (e, net) => { lock (_lock) _snapshots.Add(new KeyValuePair<int, TinyNet>(e, net)); Version++; },
                        () => _cancel);
                    Seconds = (float)sw.Elapsed.TotalSeconds;
                    Status = _cancel ? $"Stopped at epoch {Epoch}." : $"Done: {cfg.Epochs} epochs in {Seconds:F1} s, loss {LastLoss:F6}.";
                }
                catch (Exception ex) { Status = "Training failed: " + ex.Message; }
                finally { Running = false; Version++; }
            }) { IsBackground = true, Name = "CCN training: " + Title };
            _thread.Start();
        }

        public void Cancel() { _cancel = true; }

        /// <summary>Stops the worker (used when the demo is destroyed or play mode ends).</summary>
        public void Shutdown()
        {
            _cancel = true;
            if (_thread != null && _thread.IsAlive) _thread.Join(500);
        }

        // ---------- saving and loading ----------

        /// <summary>
        /// In the Editor: Assets/StreamingAssets/CCN/&lt;file&gt; (so it ships with builds).
        /// In a build: Application.persistentDataPath/&lt;file&gt;.
        /// </summary>
        public string SavePath =>
            Application.isEditor
                ? Path.Combine(Application.streamingAssetsPath, "CCN", FileName)
                : Path.Combine(Application.persistentDataPath, FileName);

        public bool Save(out string message)
        {
            TinyNet last;
            int epoch;
            lock (_lock)
            {
                if (_snapshots.Count == 0) { message = "Nothing to save yet."; return false; }
                last = _snapshots[_snapshots.Count - 1].Value;
                epoch = _snapshots[_snapshots.Count - 1].Key;
            }
            try
            {
                string path = SavePath;
                Directory.CreateDirectory(Path.GetDirectoryName(path));
                var j = last.ToJson(Title + " (trained in Unity)", $"epoch {epoch}, loss {LastLoss:G4}, {Config.Optimizer} lr {Config.LearningRate}");
                File.WriteAllText(path, JsonUtility.ToJson(j, true));
                message = "Saved " + path;
                return true;
            }
            catch (Exception e) { message = "Save failed: " + e.Message; return false; }
        }

        /// <summary>Loads a previously saved Unity-trained network, if there is one, as a single snapshot.</summary>
        public bool TryLoad(int expectInputs, int expectOutputs)
        {
            foreach (var path in new[] { SavePath, Path.Combine(Application.streamingAssetsPath, "CCN", FileName) })
            {
                try
                {
                    if (!File.Exists(path)) continue;
                    var net = TinyNet.FromJson(JsonUtility.FromJson<NetJson>(File.ReadAllText(path)));
                    if (net.InputSize != expectInputs || net.OutputSize != expectOutputs) continue;
                    lock (_lock) { _snapshots.Clear(); _snapshots.Add(new KeyValuePair<int, TinyNet>(Config.Epochs, net)); }
                    FromFile = true;
                    Status = "Loaded the saved Unity-trained network (" + FileName + "). Train again for the snapshots.";
                    Version++;
                    return true;
                }
                catch (Exception e) { Debug.LogWarning($"[CCN] Could not load {path}: {e.Message}"); }
            }
            return false;
        }
    }
}
