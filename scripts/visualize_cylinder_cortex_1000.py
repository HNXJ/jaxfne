"""Interactive 3D visualization of the default canonical cortical column
(1000 neurons, E/PV/SST/VIP) placed in a cylinder: X/Y radius 0.1mm,
Z height 1.0mm. Writes scripts/out/cortex_1000_cylinder_3d.html.
"""
import jaxfne as jtfne

OUT = "scripts/out/cortex_1000_cylinder_3d.html"


def build_cylinder_cortex(
    n: int = 1000,
    radius_mm: float = 0.1,
    height_mm: float = 1.0,
):
    cfg = (
        jtfne.build_laminar_column("V1", n=n, ei_profile="canonical")
        .update_metadata(column_radius_mm=radius_mm, column_height_mm=height_mm)
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes", "V_m"])
        .field(domain="laminar_column", conductivity="proxy")
    )
    return jtfne.construct(cfg)


def main() -> None:
    jtfne.enable_x64()
    model = build_cylinder_cortex()
    jtfne.vis.visualize_network_3d(
        model,
        title="Default canonical cortical column (1000 neurons, cylinder R=0.1mm, Z=1.0mm)",
        coordinate_unit="mm",
        display_unit="um",
        show_layers=True,
        show_column_shells=True,
        column_shape="cylinder",
        output_html=OUT,
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
