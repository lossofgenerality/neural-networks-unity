// Computational Classnotes — Cart-pole and learned pendulum (Unity 6)
// Keyboard and mouse polling that works with the Input System package or the legacy Input Manager.
// © Computational Classnotes · CC BY-NC-SA 4.0

using UnityEngine;

namespace CCN.Views
{
    public static class KeyInput
    {
        public enum K { Space, R, Tab, C, Left, Right, D1, D2, D3, D4, P, O, V, T }

        public static bool Pressed(K k)
        {
#if ENABLE_INPUT_SYSTEM
            var kb = UnityEngine.InputSystem.Keyboard.current;
            if (kb == null) return false;
            UnityEngine.InputSystem.Controls.KeyControl c;
            switch (k)
            {
                case K.Space: c = kb.spaceKey; break;
                case K.R: c = kb.rKey; break;
                case K.Tab: c = kb.tabKey; break;
                case K.C: c = kb.cKey; break;
                case K.Left: c = kb.leftArrowKey; break;
                case K.Right: c = kb.rightArrowKey; break;
                case K.D1: c = kb.digit1Key; break;
                case K.D2: c = kb.digit2Key; break;
                case K.D3: c = kb.digit3Key; break;
                case K.D4: c = kb.digit4Key; break;
                case K.P: c = kb.pKey; break;
                case K.O: c = kb.oKey; break;
                case K.V: c = kb.vKey; break;
                default: c = kb.tKey; break;
            }
            return c.wasPressedThisFrame;
#elif ENABLE_LEGACY_INPUT_MANAGER
            KeyCode c;
            switch (k)
            {
                case K.Space: c = KeyCode.Space; break;
                case K.R: c = KeyCode.R; break;
                case K.Tab: c = KeyCode.Tab; break;
                case K.C: c = KeyCode.C; break;
                case K.Left: c = KeyCode.LeftArrow; break;
                case K.Right: c = KeyCode.RightArrow; break;
                case K.D1: c = KeyCode.Alpha1; break;
                case K.D2: c = KeyCode.Alpha2; break;
                case K.D3: c = KeyCode.Alpha3; break;
                case K.D4: c = KeyCode.Alpha4; break;
                case K.P: c = KeyCode.P; break;
                case K.O: c = KeyCode.O; break;
                case K.V: c = KeyCode.V; break;
                default: c = KeyCode.T; break;
            }
            return Input.GetKeyDown(c);
#else
            return false;
#endif
        }
    }

    public static class MouseInput
    {
        /// <summary>Pointer position in screen pixels, origin bottom-left.</summary>
        public static Vector2 Position
        {
            get
            {
#if ENABLE_INPUT_SYSTEM
                var m = UnityEngine.InputSystem.Mouse.current;
                return m == null ? Vector2.zero : m.position.ReadValue();
#elif ENABLE_LEGACY_INPUT_MANAGER
                return Input.mousePosition;
#else
                return Vector2.zero;
#endif
            }
        }

        /// <summary>Left or right button held.</summary>
        public static bool DragHeld
        {
            get
            {
#if ENABLE_INPUT_SYSTEM
                var m = UnityEngine.InputSystem.Mouse.current;
                return m != null && (m.leftButton.isPressed || m.rightButton.isPressed);
#elif ENABLE_LEGACY_INPUT_MANAGER
                return Input.GetMouseButton(0) || Input.GetMouseButton(1);
#else
                return false;
#endif
            }
        }

        /// <summary>Scroll this frame in notches: positive = wheel forward (zoom in).</summary>
        public static float ScrollNotches
        {
            get
            {
#if ENABLE_INPUT_SYSTEM
                var m = UnityEngine.InputSystem.Mouse.current;
                if (m == null) return 0;
                float y = m.scroll.ReadValue().y;
                return Mathf.Abs(y) > 10f ? y / 120f : y;   // Windows reports 120 per notch
#elif ENABLE_LEGACY_INPUT_MANAGER
                return Input.mouseScrollDelta.y;
#else
                return 0;
#endif
            }
        }
    }
}
