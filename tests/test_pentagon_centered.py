from polygrid.builders import build_pentagon_centered_grid


def test_pentagon_centered_grid_center_face():
    grid = build_pentagon_centered_grid(1, embed=False)
    pent_faces = [face for face in grid.faces.values() if face.face_type == "pent"]
    assert len(pent_faces) == 1
    assert len(pent_faces[0].vertex_ids) == 5
    assert all(len(face.vertex_ids) in (5, 6) for face in grid.faces.values())
