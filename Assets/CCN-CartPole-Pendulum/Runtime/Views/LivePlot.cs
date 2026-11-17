// Computational Classnotes — Cart-pole and learned pendulum (Unity 6)
// A small live plot of θ(t), drawn into a texture and shown with IMGUI.
// © Computational Classnotes · CC BY-NC-SA 4.0

using UnityEngine;

namespace CCN.Views
{
    public sealed class LivePlot
    {
        public readonly Texture2D Texture;
        readonly int _w, _h, _capacity;
        readonly float _yMin, _yMax;
        readonly float[][] _data;
        readonly Color32[] _colors;
        readonly Color32[] _pixels;
        int _count, _head;
        bool _dirty = true;

        static readonly Color32 Bg = new Color32(22, 24, 31, 255);
        static readonly Color32 Axis = new Color32(96, 102, 116, 255);
        static readonly Color32 Grid = new Color32(40, 44, 54, 255);

        public LivePlot(int width, int height, int capacity, float yMin, float yMax, params Color[] seriesColors)
        {
            _w = width; _h = height; _capacity = capacity; _yMin = yMin; _yMax = yMax;
            _data = new float[seriesColors.Length][];
            _colors = new Color32[seriesColors.Length];
            for (int i = 0; i < seriesColors.Length; i++) { _data[i] = new float[capacity]; _colors[i] = seriesColors[i]; }
            _pixels = new Color32[width * height];
            Texture = new Texture2D(width, height, TextureFormat.RGBA32, false) { filterMode = FilterMode.Bilinear, wrapMode = TextureWrapMode.Clamp };
        }

        public void Clear() { _count = 0; _head = 0; _dirty = true; }

        public void Add(params float[] values)
        {
            for (int s = 0; s < _data.Length && s < values.Length; s++) _data[s][_head] = values[s];
            _head = (_head + 1) % _capacity;
            if (_count < _capacity) _count++;
            _dirty = true;
        }

        int Row(float y) => Mathf.Clamp(Mathf.RoundToInt((y - _yMin) / (_yMax - _yMin) * (_h - 1)), 0, _h - 1);

        void Put(int x, int y, Color32 c) { if (x >= 0 && x < _w && y >= 0 && y < _h) _pixels[y * _w + x] = c; }

        void Line(int x0, int y0, int x1, int y1, Color32 c)
        {
            int dx = Mathf.Abs(x1 - x0), dy = -Mathf.Abs(y1 - y0), sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1, err = dx + dy;
            while (true)
            {
                Put(x0, y0, c); Put(x0, y0 + 1, c);
                if (x0 == x1 && y0 == y1) break;
                int e2 = 2 * err;
                if (e2 >= dy) { err += dy; x0 += sx; }
                if (e2 <= dx) { err += dx; y0 += sy; }
            }
        }

        /// <summary>Redraws the texture if new data arrived. Call once per frame before drawing.</summary>
        public void Refresh()
        {
            if (!_dirty) return;
            _dirty = false;
            for (int i = 0; i < _pixels.Length; i++) _pixels[i] = Bg;
            for (int k = 1; k < 4; k++) { int y = _h * k / 4; for (int x = 0; x < _w; x++) Put(x, y, Grid); }
            int y0 = Row(0); for (int x = 0; x < _w; x++) Put(x, y0, Axis);

            if (_count > 1)
            {
                int start = (_head - _count + _capacity) % _capacity;
                for (int s = 0; s < _data.Length; s++)
                {
                    int px = -1, py = -1;
                    for (int i = 0; i < _count; i++)
                    {
                        float v = _data[s][(start + i) % _capacity];
                        int x = Mathf.RoundToInt(i * (_w - 1f) / (_capacity - 1f));
                        int y = Row(v);
                        if (px >= 0) Line(px, py, x, y, _colors[s]);
                        px = x; py = y;
                    }
                }
            }
            Texture.SetPixels32(_pixels);
            Texture.Apply(false);
        }
    }
}
