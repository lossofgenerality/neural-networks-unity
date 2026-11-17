// Computational Classnotes — Cart-pole and learned pendulum (Unity 6)
// Draws the cart-pole. Rail along Unity's x axis; the stick rotates about z by −θ.
// © Computational Classnotes · CC BY-NC-SA 4.0

using CCN.Core;
using UnityEngine;

namespace CCN.Views
{
    public sealed class CartPoleView
    {
        public readonly Transform Root;
        readonly Transform _cart, _stickPivot, _arrow, _arrowShaft, _arrowHead;
        readonly GameObject _stick, _tip;
        Color _stickColor = Palette.Orange;
        bool _shownFallen;

        const float CartW = 0.5f, CartH = 0.2f, CartD = 0.3f, CartBottom = 0.07f;
        const float RailY = 0.05f;
        const float ArrowMetersPerNewton = 0.06f;   // 10 N → 0.6 m

        public CartPoleView(Transform parent)
        {
            Root = Shapes.Empty("Cart-pole", parent, Vector3.zero);

            Shapes.GridFloor(Root, 40f);

            // Rail (±3 m) on two supports, end stops and metre marks.
            Shapes.Make(PrimitiveType.Cube, "Rail", Root, new Vector3(0, RailY, 0), new Vector3(6.1f, 0.025f, 0.05f), Palette.Metal, 0.8f, 0.9f);
            Shapes.Make(PrimitiveType.Cube, "Rail base", Root, new Vector3(0, 0.015f, 0), new Vector3(6.4f, 0.03f, 0.22f), Palette.DarkMetal, 0.35f, 0.2f);
            for (int i = -3; i <= 3; i++)
            {
                bool end = i == -3 || i == 3;
                if (end)
                    Shapes.Make(PrimitiveType.Cube, "Stop", Root, new Vector3(i * 1.0167f, 0.11f, 0), new Vector3(0.05f, 0.16f, 0.2f), Palette.Force, 0.4f, 0f);
                else
                    Shapes.Make(PrimitiveType.Cube, "Mark " + i + " m", Root, new Vector3(i, 0.031f, -0.09f), new Vector3(i == 0 ? 0.02f : 0.012f, 0.004f, 0.04f), Palette.Gray, 0.3f, 0f);
            }

            // Cart with the stick pivot on its top face.
            _cart = Shapes.Empty("Cart", Root, Vector3.zero);
            Shapes.Make(PrimitiveType.Cube, "Body", _cart, new Vector3(0, CartBottom + CartH / 2, 0), new Vector3(CartW, CartH, CartD), Palette.Cart, 0.6f, 0.1f);
            Shapes.Make(PrimitiveType.Cube, "Top plate", _cart, new Vector3(0, CartBottom + CartH + 0.005f, 0), new Vector3(CartW * 0.92f, 0.01f, CartD * 0.92f), Palette.DarkMetal, 0.7f, 0.6f);
            foreach (float wx in new[] { -0.17f, 0.17f })
                foreach (float wz in new[] { -0.155f, 0.155f })
                {
                    var w = Shapes.Make(PrimitiveType.Cylinder, "Wheel", _cart, new Vector3(wx, 0.075f, wz), new Vector3(0.09f, 0.012f, 0.09f), Palette.DarkMetal, 0.5f, 0.3f);
                    w.transform.localRotation = Quaternion.Euler(90, 0, 0);
                }

            _stickPivot = Shapes.Empty("Stick pivot", _cart, new Vector3(0, CartBottom + CartH + 0.01f, 0));
            var hinge = Shapes.Make(PrimitiveType.Cylinder, "Hinge", _stickPivot, Vector3.zero, new Vector3(0.06f, 0.05f, 0.06f), Palette.Metal, 0.8f, 0.9f);
            hinge.transform.localRotation = Quaternion.Euler(90, 0, 0);
            // Cylinder primitive is 2 units tall: scale y 0.5 → 1 m stick, shifted up by 0.5 so it hinges at its end.
            _stick = Shapes.Make(PrimitiveType.Cylinder, "Stick (1 m)", _stickPivot, new Vector3(0, 0.5f, 0), new Vector3(0.035f, 0.5f, 0.035f), Palette.Orange, 0.55f, 0f);
            _tip = Shapes.Make(PrimitiveType.Sphere, "Tip", _stickPivot, new Vector3(0, 1f, 0), Vector3.one * 0.06f, Palette.Orange, 0.55f, 0f);

            // Force arrow in front of the cart (towards the camera).
            _arrow = Shapes.Empty("Force arrow", _cart, new Vector3(0, CartBottom + CartH / 2, -CartD / 2 - 0.07f));
            _arrowShaft = Shapes.Make(PrimitiveType.Cube, "Shaft", _arrow, Vector3.zero, new Vector3(0.1f, 0.035f, 0.035f), Palette.Force, 0.4f, 0f).transform;
            _arrowHead = Shapes.Make(PrimitiveType.Cube, "Head", _arrow, Vector3.zero, new Vector3(0.07f, 0.07f, 0.035f), Palette.Force, 0.4f, 0f).transform;
            _arrowHead.localRotation = Quaternion.Euler(0, 0, 45);
        }

        public void SetActive(bool on) => Root.gameObject.SetActive(on);

        /// <summary>Stick colour shows who is in control (orange notebook net, teal Unity-trained net, …).</summary>
        public void SetStickColor(Color c)
        {
            _stickColor = c;
            if (!_shownFallen) { Shapes.SetColor(_stick, c); Shapes.SetColor(_tip, c); }
        }

        public void Show(CartPole cp, bool showForce)
        {
            var s = cp.State;
            _cart.localPosition = new Vector3((float)s.X, 0, 0);

            // Fallen: freeze lying down at ±90°.
            float th = Mathf.Clamp((float)s.Theta, -Mathf.PI / 2, Mathf.PI / 2);
            _stickPivot.localRotation = Quaternion.Euler(0, 0, -th * Mathf.Rad2Deg);

            if (cp.Fallen != _shownFallen)
            {
                _shownFallen = cp.Fallen;
                var c = cp.Fallen ? Palette.Fallen : _stickColor;
                Shapes.SetColor(_stick, c); Shapes.SetColor(_tip, c);
            }

            // Arrow: length ∝ u, pointing in the direction of the force, starting at the cart's centre line.
            float u = (float)cp.LastForce;
            float len = Mathf.Abs(u) * ArrowMetersPerNewton;
            bool visible = showForce && !cp.Fallen && len > 0.005f;
            _arrow.gameObject.SetActive(visible);
            if (visible)
            {
                float dir = Mathf.Sign(u);
                _arrowShaft.localScale = new Vector3(len, 0.035f, 0.035f);
                _arrowShaft.localPosition = new Vector3(dir * len / 2, 0, 0);
                _arrowHead.localPosition = new Vector3(dir * len, 0, 0);
            }
        }
    }
}
