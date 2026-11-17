// Computational Classnotes — Cart-pole and learned pendulum (Unity 6)
// A 3D camera that orbits and zooms around the demo, within limits: no panning, no going
// under the floor, no turning round to the back.
// © Computational Classnotes · CC BY-NC-SA 4.0

using UnityEngine;

namespace CCN.Views
{
    public sealed class OrbitCamera
    {
        public readonly Camera Cam;

        // Per-scene framing and limits.
        Vector3 _target;
        float _homeYaw, _homePitch, _homeDist;
        float _minDist, _maxDist, _maxYaw, _maxPitch;
        const float FloorClearance = 0.12f;

        float _yaw, _pitch, _dist;          // current (smoothed)
        float _toYaw, _toPitch, _toDist;    // wanted
        Vector2 _lastPointer;
        bool _dragging, _dragBlocked, _wasHeld;

        public float DegreesPerPixel = 0.25f;
        public float ZoomPerNotch = 0.12f;
        public float Smoothing = 12f;

        public OrbitCamera(Camera cam) { Cam = cam; }

        /// <summary>Sets where the camera looks and how far it may move. yaw 0 = from the front (−z).</summary>
        public void Frame(Vector3 target, float yaw, float pitch, float dist,
                          float minDist, float maxDist, float maxYaw, float maxPitch, bool snap)
        {
            _target = target;
            _homeYaw = yaw; _homePitch = pitch; _homeDist = dist;
            _minDist = minDist; _maxDist = maxDist; _maxYaw = maxYaw; _maxPitch = maxPitch;
            ResetView(snap);
        }

        public void ResetView(bool snap = false)
        {
            _toYaw = _homeYaw; _toPitch = _homePitch; _toDist = _homeDist;
            if (snap) { _yaw = _toYaw; _pitch = _toPitch; _dist = _toDist; Apply(); }
        }

        /// <summary>Lowest pitch (deg) that keeps the camera above the floor at distance d.</summary>
        float MinPitch(float d) => Mathf.Asin(Mathf.Clamp((FloorClearance - _target.y) / d, -1f, 1f)) * Mathf.Rad2Deg;

        /// <param name="pointerOverUi">True when the pointer is on the control panel; drags that start there are ignored.</param>
        public void Tick(float dt, bool pointerOverUi)
        {
            Vector2 p = MouseInput.Position;
            bool held = MouseInput.DragHeld;
            if (held && !_wasHeld) { _dragging = true; _dragBlocked = pointerOverUi; _lastPointer = p; }
            if (!held) _dragging = false;
            _wasHeld = held;

            if (_dragging && !_dragBlocked)
            {
                Vector2 d = p - _lastPointer;
                _toYaw += d.x * DegreesPerPixel;
                _toPitch -= d.y * DegreesPerPixel;
            }
            _lastPointer = p;

            if (!pointerOverUi)
            {
                float n = MouseInput.ScrollNotches;
                if (Mathf.Abs(n) > 0.001f) _toDist *= Mathf.Pow(1f - ZoomPerNotch, n);
            }

            _toDist = Mathf.Clamp(_toDist, _minDist, _maxDist);
            _toYaw = Mathf.Clamp(_toYaw, -_maxYaw, _maxYaw);
            _toPitch = Mathf.Clamp(_toPitch, MinPitch(_toDist), _maxPitch);

            float k = 1f - Mathf.Exp(-Smoothing * dt);
            _yaw = Mathf.Lerp(_yaw, _toYaw, k);
            _pitch = Mathf.Lerp(_pitch, _toPitch, k);
            _dist = Mathf.Lerp(_dist, _toDist, k);
            Apply();
        }

        void Apply()
        {
            var rot = Quaternion.Euler(_pitch, _yaw, 0);
            Cam.transform.SetPositionAndRotation(_target + rot * new Vector3(0, 0, -_dist), rot);
        }
    }
}
