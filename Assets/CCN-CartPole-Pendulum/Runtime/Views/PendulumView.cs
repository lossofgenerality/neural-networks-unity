// Computational Classnotes — Cart-pole and learned pendulum (Unity 6)
// Draws the two pendulums. The bob is at (sin θ, −cos θ) from the pivot in the x-y plane.
// © Computational Classnotes · CC BY-NC-SA 4.0

using UnityEngine;

namespace CCN.Views
{
    /// <summary>One pendulum: pivot, 1-unit rod, bob.</summary>
    public sealed class PendulumRig
    {
        public readonly Transform Pivot;
        readonly Transform _arm;
        readonly GameObject _bob;

        public PendulumRig(string name, Transform parent, Color color, float bobRadius)
        {
            Pivot = Shapes.Empty(name, parent, Vector3.zero);
            var pin = Shapes.Make(PrimitiveType.Cylinder, "Pivot pin", Pivot, Vector3.zero, new Vector3(0.06f, 0.06f, 0.06f), Palette.Metal, 0.8f, 0.9f);
            pin.transform.localRotation = Quaternion.Euler(90, 0, 0);
            _arm = Shapes.Empty("Arm", Pivot, Vector3.zero);
            Shapes.Make(PrimitiveType.Cylinder, "Rod", _arm, new Vector3(0, -0.5f, 0), new Vector3(0.025f, 0.5f, 0.025f), Palette.Metal, 0.75f, 0.8f);
            _bob = Shapes.Make(PrimitiveType.Sphere, "Bob", _arm, new Vector3(0, -1f, 0), Vector3.one * (2 * bobRadius), color, 0.6f, 0.05f);
        }

        public void SetColor(Color c) => Shapes.SetColor(_bob, c);

        /// <summary>Rotating the downward arm about z by +θ puts the bob at (sin θ, −cos θ).</summary>
        public void SetAngle(double theta) => _arm.localRotation = Quaternion.Euler(0, 0, (float)(theta * Mathf.Rad2Deg));
    }

    public sealed class PendulumView
    {
        public readonly Transform Root;
        public readonly PendulumRig TrueRig, NetRig;
        public const float PivotY = 1.35f;
        const float SideX = 1.3f;

        public PendulumView(Transform parent)
        {
            Root = Shapes.Empty("Learned pendulum", parent, Vector3.zero);
            Shapes.GridFloor(Root, 40f);

            // Gantry: beam on two posts with feet.
            Shapes.Make(PrimitiveType.Cube, "Beam", Root, new Vector3(0, PivotY + 0.05f, 0.06f), new Vector3(4.4f, 0.06f, 0.06f), Palette.DarkMetal, 0.6f, 0.7f);
            foreach (float x in new[] { -2.2f, 2.2f })
            {
                Shapes.Make(PrimitiveType.Cube, "Post", Root, new Vector3(x, (PivotY + 0.08f) / 2, 0.06f), new Vector3(0.06f, PivotY + 0.08f, 0.06f), Palette.DarkMetal, 0.6f, 0.7f);
                Shapes.Make(PrimitiveType.Cube, "Foot", Root, new Vector3(x, 0.015f, 0.06f), new Vector3(0.12f, 0.03f, 0.6f), Palette.DarkMetal, 0.5f, 0.5f);
            }

            TrueRig = new PendulumRig("True pendulum (gray)", Root, Palette.Gray, 0.10f);
            NetRig = new PendulumRig("Network pendulum", Root, Palette.Orange, 0.09f);
            SetLayout(false);
        }

        public void SetActive(bool on) => Root.gameObject.SetActive(on);

        /// <summary>Side by side, or both from the same pivot (gray one slightly behind and larger).</summary>
        public void SetLayout(bool samePivot)
        {
            if (samePivot)
            {
                TrueRig.Pivot.localPosition = new Vector3(0, PivotY, 0.0f);
                NetRig.Pivot.localPosition = new Vector3(0, PivotY, -0.12f);
            }
            else
            {
                TrueRig.Pivot.localPosition = new Vector3(-SideX, PivotY, 0);
                NetRig.Pivot.localPosition = new Vector3(SideX, PivotY, 0);
            }
        }

        public void Show(double thetaTrue, double thetaNet)
        {
            TrueRig.SetAngle(thetaTrue);
            NetRig.SetAngle(thetaNet);
        }
    }
}
