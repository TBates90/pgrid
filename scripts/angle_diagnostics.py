from __future__ import annotations

import math
from collections import defaultdict

from polygrid.builders import build_pentagon_centered_grid
from polygrid.algorithms import build_face_adjacency, ring_faces


def angle_at(a, b, c) -> float:
    v1x, v1y = a.x - b.x, a.y - b.y
    v2x, v2y = c.x - b.x, c.y - b.y
    denom = (math.hypot(v1x, v1y) * math.hypot(v2x, v2y)) or 1.0
    dot = (v1x * v2x + v1y * v2y) / denom
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


def main() -> None:
    grid = build_pentagon_centered_grid(2, embed=True, embed_mode="angle")
    pent = next(face for face in grid.faces.values() if face.face_type == "pent")
    adjacency = build_face_adjacency(grid.faces.values(), grid.edges.values())
    rings = ring_faces(adjacency, pent.id, max_depth=3)

    for ring_idx, face_ids in rings.items():
        if ring_idx == 0:
            continue
        angles = []
        edge_lengths = []
        for fid in face_ids:
            face = grid.faces[fid]
            verts = list(face.vertex_ids)
            if len(verts) != 6:
                continue
            xs = [grid.vertices[vid].x for vid in verts]
            ys = [grid.vertices[vid].y for vid in verts]
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            verts = sorted(verts, key=lambda vid: math.atan2(grid.vertices[vid].y - cy, grid.vertices[vid].x - cx))
            for i, vid in enumerate(verts):
                prev_vid = verts[(i - 1) % len(verts)]
                next_vid = verts[(i + 1) % len(verts)]
                angles.append(
                    angle_at(
                        grid.vertices[prev_vid],
                        grid.vertices[vid],
                        grid.vertices[next_vid],
                    )
                )
            for i in range(len(verts)):
                a = grid.vertices[verts[i]]
                b = grid.vertices[verts[(i + 1) % len(verts)]]
                edge_lengths.append(math.hypot(b.x - a.x, b.y - a.y))

        if not angles:
            continue
        print(
            f"Ring {ring_idx}: angle min/max {min(angles):.3f}/{max(angles):.3f}, "
            f"edge min/max {min(edge_lengths):.3f}/{max(edge_lengths):.3f}"
        )


if __name__ == "__main__":
    main()
