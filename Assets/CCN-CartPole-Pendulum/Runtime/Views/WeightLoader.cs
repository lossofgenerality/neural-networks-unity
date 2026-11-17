// Computational Classnotes — Cart-pole and learned pendulum (Unity 6)
// Loads network weights from StreamingAssets/CCN/*.json, falling back to the built-in weights.
// © Computational Classnotes · CC BY-NC-SA 4.0

using System;
using System.IO;
using CCN.Core;
using UnityEngine;

namespace CCN.Views
{
    public static class WeightLoader
    {
        /// <summary>
        /// Reads StreamingAssets/CCN/&lt;fileName&gt;. If it is missing, unreadable (e.g. Android/WebGL, where
        /// StreamingAssets is not a plain folder) or has the wrong shape, returns the fallback.
        /// </summary>
        public static TinyNet Load(string fileName, Func<TinyNet> fallback, int expectInputs, int expectOutputs, out string source)
        {
            string path = Path.Combine(Application.streamingAssetsPath, "CCN", fileName);
            try
            {
                if (File.Exists(path))
                {
                    var j = JsonUtility.FromJson<NetJson>(File.ReadAllText(path));
                    var net = TinyNet.FromJson(j);
                    if (net.InputSize != expectInputs || net.OutputSize != expectOutputs)
                        throw new Exception($"expected {expectInputs} inputs and {expectOutputs} outputs, got {net.InputSize} and {net.OutputSize}");
                    source = "JSON: " + fileName;
                    return net;
                }
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[CCN] Could not use {path} ({e.Message}); using the built-in weights.");
            }
            source = "built-in";
            return fallback();
        }
    }
}
