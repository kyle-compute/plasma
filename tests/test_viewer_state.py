from __future__ import annotations

import numpy as np

from plasma.live.contracts import LiveField2D, LiveParticleCloud, LiveSnapshot
from plasma.viewer.state import emissive_points, particle_ring_points


def test_emissive_points_apply_phase_offset_between_axial_cells():
    snapshot = LiveSnapshot(
        model="pic",
        state="running",
        title="demo",
        updated_at_s=0.0,
        fields={
            "emissivity_arb": LiveField2D(
                x=[0.0, 0.4],
                y=[0.2],
                values=[[1.0, 1.0]],
            )
        },
    )

    points, intensity = emissive_points(snapshot, n_theta=4, threshold=0.0)

    assert points.shape == (8, 3)
    assert intensity.shape == (8,)
    assert not np.allclose(points[:4, :2], points[4:, :2])


def test_particle_ring_points_preserve_axial_positions():
    cloud = LiveParticleCloud(r=[0.2, 0.2, 0.2], z=[0.0, 0.4, 0.8], energy_ev=[1.0, 2.0, 3.0])

    points = particle_ring_points(cloud, theta_offset=0.25)

    assert points.shape == (3, 3)
    assert np.allclose(points[:, 2], np.asarray(cloud.z))
