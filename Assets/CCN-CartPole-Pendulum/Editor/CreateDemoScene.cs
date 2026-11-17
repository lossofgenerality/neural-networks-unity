// Computational Classnotes — Cart-pole and learned pendulum (Unity 6)
// Editor menu: builds and saves the demo scene, and writes the weight JSON files.
// © Computational Classnotes · CC BY-NC-SA 4.0

using System.IO;
using System.Linq;
using CCN.Core;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

static class CreateDemoScene
{
    const string ScenePath = "Assets/CCN-CartPole-Pendulum/Scenes/CartPolePendulum.unity";

    [MenuItem("Computational Classnotes/Create Cart-Pole & Pendulum Scene")]
    static void Create()
    {
        if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
        var scene = EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);
        new GameObject("CCN Demo").AddComponent<CCNDemo>();

        Directory.CreateDirectory(Path.GetDirectoryName(ScenePath));
        EditorSceneManager.SaveScene(scene, ScenePath);

        var list = EditorBuildSettings.scenes.ToList();
        if (!list.Any(s => s.path == ScenePath))
        {
            list.Insert(0, new EditorBuildSettingsScene(ScenePath, true));
            EditorBuildSettings.scenes = list.ToArray();
        }
        Debug.Log($"[CCN] Saved {ScenePath} and added it to the build. Press Play.");
    }

    [MenuItem("Computational Classnotes/Write Built-in Weights to StreamingAssets")]
    static void WriteWeights()
    {
        string dir = Path.Combine(Application.dataPath, "StreamingAssets", "CCN");
        Directory.CreateDirectory(dir);
        File.WriteAllText(Path.Combine(dir, "cartpole_net.json"),
            JsonUtility.ToJson(TrainedWeights.CartPoleNet().ToJson("cartpole 4-8-1", "u = clamp(10*out, -10, 10)"), true));
        File.WriteAllText(Path.Combine(dir, "pendulum_net.json"),
            JsonUtility.ToJson(TrainedWeights.PendulumNet().ToJson("pendulum 1-6-1", "a(theta) ~ -sin(theta)"), true));
        AssetDatabase.Refresh();
        Debug.Log("[CCN] Wrote cartpole_net.json and pendulum_net.json to Assets/StreamingAssets/CCN.");
    }
}
