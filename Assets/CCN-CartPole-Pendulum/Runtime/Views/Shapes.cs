// Computational Classnotes — Cart-pole and learned pendulum (Unity 6)
// Helpers for building the scene from primitives (works in URP and the built-in pipeline),
// and the dark studio look: background, grid floor, fog, shadows, ambient light.
// © Computational Classnotes · CC BY-NC-SA 4.0

using UnityEngine;
using UnityEngine.Rendering;

namespace CCN.Views
{
    public static class Palette
    {
        public static readonly Color Orange     = new Color(0.96f, 0.52f, 0.16f);
        public static readonly Color Gray       = new Color(0.72f, 0.73f, 0.76f);
        public static readonly Color Teal       = new Color(0.20f, 0.74f, 0.70f);   // Unity-trained network
        public static readonly Color Metal      = new Color(0.50f, 0.52f, 0.57f);
        public static readonly Color DarkMetal  = new Color(0.26f, 0.27f, 0.31f);
        public static readonly Color Cart       = new Color(0.20f, 0.42f, 0.78f);
        public static readonly Color Force      = new Color(0.92f, 0.26f, 0.24f);
        public static readonly Color Fallen     = new Color(0.80f, 0.22f, 0.22f);
        public static readonly Color Background = new Color(0.075f, 0.082f, 0.105f);
        public static readonly Color Floor      = new Color(0.135f, 0.145f, 0.175f);
        public static readonly Color GridLine   = new Color(0.24f, 0.26f, 0.31f);
        public static readonly Color GridMajor  = new Color(0.33f, 0.36f, 0.42f);
    }

    public static class Shapes
    {
        /// <summary>Creates a primitive without a collider (our physics is our own), coloured, under a parent.</summary>
        public static GameObject Make(PrimitiveType type, string name, Transform parent, Vector3 localPos, Vector3 localScale, Color color,
                                      float smoothness = 0.45f, float metallic = 0f)
        {
            var go = GameObject.CreatePrimitive(type);
            go.name = name;
            var col = go.GetComponent<Collider>();
            if (col != null) Object.Destroy(col);
            go.transform.SetParent(parent, false);
            go.transform.localPosition = localPos;
            go.transform.localScale = localScale;
            SetColor(go, color);
            SetSurface(go, smoothness, metallic);
            return go;
        }

        /// <summary>Material.color maps to the shader's [MainColor] (_BaseColor in URP Lit, _Color in Standard).</summary>
        public static void SetColor(GameObject go, Color c)
        {
            var r = go.GetComponent<Renderer>();
            if (r != null) r.material.color = c;
        }

        static readonly int Smoothness = Shader.PropertyToID("_Smoothness");   // URP Lit
        static readonly int Glossiness = Shader.PropertyToID("_Glossiness");   // built-in Standard
        static readonly int Metallic = Shader.PropertyToID("_Metallic");

        public static void SetSurface(GameObject go, float smoothness, float metallic)
        {
            var r = go.GetComponent<Renderer>();
            if (r == null) return;
            var m = r.material;
            if (m.HasProperty(Smoothness)) m.SetFloat(Smoothness, smoothness);
            if (m.HasProperty(Glossiness)) m.SetFloat(Glossiness, smoothness);
            if (m.HasProperty(Metallic)) m.SetFloat(Metallic, metallic);
        }

        public static Transform Empty(string name, Transform parent, Vector3 localPos)
        {
            var t = new GameObject(name).transform;
            t.SetParent(parent, false);
            t.localPosition = localPos;
            return t;
        }

        /// <summary>A large floor with a 1 m grid (brighter line every metre, faint every 25 cm); fog fades its edges.</summary>
        public static GameObject GridFloor(Transform parent, float size)
        {
            var go = Make(PrimitiveType.Plane, "Floor (grid)", parent, Vector3.zero, new Vector3(size / 10f, 1, size / 10f), Color.white, 0.25f, 0f);
            var r = go.GetComponent<Renderer>();
            r.material.mainTexture = GridTexture();
            r.material.mainTextureScale = new Vector2(size, size);   // one texture tile per metre
            r.shadowCastingMode = ShadowCastingMode.Off;
            r.receiveShadows = true;
            return go;
        }

        static Texture2D _grid;
        static Texture2D GridTexture()
        {
            if (_grid != null) return _grid;
            const int n = 256;
            _grid = new Texture2D(n, n, TextureFormat.RGBA32, true) { wrapMode = TextureWrapMode.Repeat, filterMode = FilterMode.Trilinear, anisoLevel = 8, name = "CCN grid" };
            var px = new Color[n * n];
            for (int y = 0; y < n; y++)
                for (int x = 0; x < n; x++)
                {
                    bool major = x < 3 || y < 3;
                    bool minor = (x % (n / 4)) < 2 || (y % (n / 4)) < 2;
                    px[y * n + x] = major ? Palette.GridMajor : minor ? Palette.GridLine : Palette.Floor;
                }
            _grid.SetPixels(px);
            _grid.Apply(true);
            return _grid;
        }

        /// <summary>Dark background, fog matched to it, soft-shadowed key light, a cool rim light and trilight ambient.</summary>
        public static void StudioLighting(Camera cam)
        {
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = Palette.Background;
            cam.allowHDR = true;
            cam.allowMSAA = true;

            RenderSettings.skybox = null;
            RenderSettings.fog = true;
            RenderSettings.fogMode = FogMode.Linear;
            RenderSettings.fogColor = Palette.Background;
            RenderSettings.fogStartDistance = 7f;
            RenderSettings.fogEndDistance = 18f;
            RenderSettings.ambientMode = AmbientMode.Trilight;
            RenderSettings.ambientSkyColor = new Color(0.30f, 0.33f, 0.40f);
            RenderSettings.ambientEquatorColor = new Color(0.17f, 0.18f, 0.22f);
            RenderSettings.ambientGroundColor = new Color(0.07f, 0.07f, 0.09f);

            Light key = null;
            foreach (var l in Object.FindObjectsByType<Light>(FindObjectsSortMode.None))
                if (l.type == LightType.Directional) { key = l; break; }
            if (key == null) key = new GameObject("Key Light").AddComponent<Light>();
            key.type = LightType.Directional;
            key.color = new Color(1f, 0.96f, 0.90f);
            key.intensity = 1.25f;
            key.shadows = LightShadows.Soft;
            key.shadowStrength = 0.75f;
            key.transform.rotation = Quaternion.Euler(48, -35, 0);
            RenderSettings.sun = key;

            var rim = new GameObject("Rim Light").AddComponent<Light>();
            rim.type = LightType.Directional;
            rim.color = new Color(0.55f, 0.68f, 1f);
            rim.intensity = 0.35f;
            rim.shadows = LightShadows.None;
            rim.transform.rotation = Quaternion.Euler(25, 150, 0);
        }
    }
}
