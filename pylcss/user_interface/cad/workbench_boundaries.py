# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""WorkbenchBoundaryMixin behavior for the Design Studio workbench."""

from __future__ import annotations

import logging

import numpy as np


from .boundary_visuals import direction_from_label

logger = logging.getLogger(__name__)

__all__ = ["WorkbenchBoundaryMixin"]


class WorkbenchBoundaryMixin:
    @staticmethod
    def _get_face_centroid(occ_face):
        """Compute the geometric center, with a bounding-box fallback."""
        if isinstance(occ_face, dict):
            center = occ_face.get("center")
            if center is not None:
                return [float(v) for v in center[:3]]
            bbox = occ_face.get("bbox") or {}
            try:
                return [
                    (float(bbox["xmin"]) + float(bbox["xmax"])) / 2.0,
                    (float(bbox["ymin"]) + float(bbox["ymax"])) / 2.0,
                    (float(bbox["zmin"]) + float(bbox["zmax"])) / 2.0,
                ]
            except Exception:
                return [0.0, 0.0, 0.0]
        try:
            center = occ_face.Center()
            return [float(center.x), float(center.y), float(center.z)]
        except Exception:
            logger.debug("Optional UI operation failed.", exc_info=True)
        try:
            bb = occ_face.BoundingBox()
            return [
                (bb.xmin + bb.xmax) / 2.0,
                (bb.ymin + bb.ymax) / 2.0,
                (bb.zmin + bb.zmax) / 2.0,
            ]
        except Exception:
            return [0.0, 0.0, 0.0]

    def _collect_bc_for_node(self, anchor_node):
        """
        Walk the entire graph and collect load/constraint BC data that feeds
        into (or is associated with) *anchor_node*.

        Returns:
            (constraint_faces, load_faces, load_vectors)
            Each is a list (possibly empty). load_vectors is a list of
            ([cx,cy,cz], [fx,fy,fz]) tuples.
        """
        constraint_faces = []
        load_faces = []
        load_vectors = []
        topology_load_face_keys = set()

        anchor_cls = anchor_node.__class__.__name__
        # Conditions shown in the viewer must belong to the selected node's
        # upstream workflow.  The previous solver-only scoping leaked loads and
        # supports from disconnected studies when selecting geometry or an
        # individual setup node.
        overlay_scope = set()

        def _walk_upstream(n):
            marker = id(n)
            if marker in overlay_scope:
                return
            overlay_scope.add(marker)
            try:
                ports = n.input_ports()
                if isinstance(ports, dict):
                    ports = list(ports.values())
            except Exception:
                ports = []
            for port in ports:
                try:
                    connected = list(port.connected_ports())
                except Exception:
                    connected = []
                for conn_port in connected:
                    try:
                        _walk_upstream(conn_port.node())
                    except Exception:
                        logger.debug("Optional UI operation failed.", exc_info=True)

        _walk_upstream(anchor_node)

        crash_moving_body = False
        if anchor_cls == "CrashSolverNode":
            for candidate in self.graph.all_nodes():
                if candidate.__class__.__name__ == "ImpactConditionNode" and (
                    id(candidate) in overlay_scope
                ):
                    scope = (
                        str(candidate.get_property("application_scope") or "")
                        .strip()
                        .lower()
                        .replace("_", " ")
                    )
                    crash_moving_body = scope.startswith("moving body")
                    break

        def _get_faces_for_bc_node(bc_node, extra_port_names=None):
            """Try two sources: (1) _last_result geometries, (2) named input port to SelectFaceNode.

            extra_port_names: additional port names to walk (e.g. ['impact_face'])
            in addition to the default 'target_face'.
            """
            faces = []
            # Source 1: cached result of BC node itself (populated after full run)
            try:
                result = getattr(bc_node, "_last_result", None)
                if isinstance(result, dict):
                    geoms = result.get("geometries") or []
                    # PressureLoadNode and older ConstraintNode use 'geometry' (singular)
                    singular_geometry = result.get("geometry")
                    if not geoms and singular_geometry is not None:
                        geoms = [singular_geometry]
                    # ImpactConditionNode stores faces under 'face_list'
                    if not geoms:
                        geoms = result.get("face_list") or []
                    faces.extend([g for g in geoms if g is not None])
            except Exception:
                logger.debug("Optional UI operation failed.", exc_info=True)

            # Source 2: walk named input ports to SelectFaceNode result
            if not faces:
                port_names_to_check = {"target_face"}
                if extra_port_names:
                    port_names_to_check.update(extra_port_names)
                try:
                    for port in bc_node.input_ports():
                        try:
                            pname = port.name()
                        except Exception:
                            pname = ""
                        if pname not in port_names_to_check:
                            continue
                        for conn_port in port.connected_ports():
                            upstream = conn_port.node()
                            upstream_result = getattr(upstream, "_last_result", None)
                            if isinstance(upstream_result, dict):
                                sel_faces = (
                                    upstream_result.get("entities")
                                    or upstream_result.get("faces")
                                    or []
                                )
                                faces.extend([f for f in sel_faces if f is not None])
                                if not faces:
                                    entity = upstream_result.get(
                                        "entity"
                                    ) or upstream_result.get("face")
                                    if entity is not None:
                                        faces.append(entity)
                except Exception:
                    logger.debug("Optional UI operation failed.", exc_info=True)

            return faces

        def _get_mesh_for_bc_node(bc_node):
            """Return the mesh connected to a BC node, if it has already run."""
            try:
                mesh_port = bc_node.get_input("mesh")
                if mesh_port:
                    for conn_port in mesh_port.connected_ports():
                        upstream = conn_port.node()
                        mesh = getattr(upstream, "_last_result", None)
                        if (
                            mesh is not None
                            and hasattr(mesh, "p")
                            and hasattr(mesh, "t")
                        ):
                            return mesh
            except Exception:
                logger.debug("Optional UI operation failed.", exc_info=True)
            return None

        def _condition_centroid_and_points(bc_node):
            """Centroid/sample points for legacy condition-expression BCs."""
            condition = ""
            try:
                condition = str(bc_node.get_property("condition") or "").strip()
            except Exception:
                condition = ""
            if not condition:
                return None, []
            mesh = _get_mesh_for_bc_node(bc_node)
            if mesh is None:
                return None, []
            try:
                import numpy as _np
                from pylcss.solver_backends.selection import nodes_matching_condition

                node_ids = nodes_matching_condition(mesh, condition)
                if node_ids.size == 0:
                    return None, []
                pts = _np.asarray(mesh.p, dtype=float).T[node_ids]
                centroid = _np.mean(pts, axis=0).tolist()
                if len(pts) <= 5:
                    samples = pts.tolist()
                else:
                    order = _np.linspace(0, len(pts) - 1, 5, dtype=int)
                    samples = pts[order].tolist()
                return centroid, samples
            except Exception:
                return None, []

        for node in self.graph.all_nodes():
            cls = node.__class__.__name__
            if (
                id(node) not in overlay_scope
            ):
                continue

            # ---- ConstraintNode ----
            if cls == "ConstraintNode":
                # The backend deliberately drops SPC constraints for a
                # free-flying moving-body barrier impact. Showing them here
                # would misrepresent the model actually sent to OpenRadioss.
                if anchor_cls == "CrashSolverNode" and crash_moving_body:
                    continue
                if anchor_cls == "ConstraintNode" and node is not anchor_node:
                    continue
                if anchor_cls == "LoadNode":
                    continue
                faces = _get_faces_for_bc_node(node)
                # FIX #V4: Carry viz metadata (per-type color) alongside the face
                # object so the viewer can render each constraint type distinctly.
                result = getattr(node, "_last_result", None)
                viz_meta = (
                    result.get("viz") if isinstance(result, dict) else None
                ) or {}
                for g in faces:
                    if isinstance(g, dict) and (
                        g.get("mesh_selection") or g.get("node_ids") is not None
                    ):
                        mesh_viz = dict(viz_meta)
                        if not mesh_viz.get("faces"):
                            mesh_viz["faces"] = [
                                {
                                    "center": g.get("center"),
                                    "points": g.get("points"),
                                    "bbox": g.get("bbox"),
                                }
                            ]
                        constraint_faces.append(
                            {
                                "pos": g.get("center"),
                                "points": g.get("points"),
                                "viz": mesh_viz,
                                "mesh_selection": g,
                            }
                        )
                    else:
                        constraint_faces.append({"face": g, "viz": viz_meta})
                if not faces:
                    centroid, samples = _condition_centroid_and_points(node)
                    if centroid is not None:
                        if not viz_meta:
                            try:
                                fixed_dofs = list(
                                    (result or {}).get("fixed_dofs") or []
                                )
                            except Exception:
                                fixed_dofs = []
                            viz_meta = {
                                "constraint_type": node.get_property("constraint_type")
                                or "Fixed",
                                "color": "#2979FF",
                                "fixed_dofs": fixed_dofs or [0, 1, 2],
                            }
                        constraint_faces.append(
                            {
                                "pos": centroid,
                                "points": samples,
                                "viz": viz_meta,
                            }
                        )

            # ---- LoadNode ----
            elif cls == "LoadNode":
                if anchor_cls == "LoadNode" and node is not anchor_node:
                    continue
                if anchor_cls == "ConstraintNode":
                    continue
                load_type = str(node.get_property("load_type") or "Force").strip()
                if load_type == "Gravity":
                    try:
                        accel = float(node.get_property("gravity_accel") or 0.0)
                        direction_label = (
                            str(node.get_property("gravity_direction") or "-Y")
                            .strip()
                            .upper()
                        )
                        direction = direction_from_label(direction_label)
                        mesh = _get_mesh_for_bc_node(node)
                        if mesh is not None:
                            mesh_points = np.asarray(mesh.p, dtype=float).T
                            center = np.mean(mesh_points, axis=0).tolist()
                        else:
                            bounds = self.viewer.current_actor.GetBounds()
                            center = [
                                0.5 * (bounds[0] + bounds[1]),
                                0.5 * (bounds[2] + bounds[3]),
                                0.5 * (bounds[4] + bounds[5]),
                            ]
                        load_vectors.append(
                            {
                                "centroid": center,
                                "vector": (direction * accel).tolist(),
                                "magnitude_N": accel,
                                "quantity_type": "gravity",
                                "label": f"g {accel / 1000.0:.4g} m/s²",
                                "color": "#B875FF",
                            }
                        )
                    except Exception:
                        logger.debug("Optional UI operation failed.", exc_info=True)
                    continue
                faces = _get_faces_for_bc_node(node)
                try:
                    fx = float(node.get_property("force_x") or 0.0)
                    fy = float(node.get_property("force_y") or 0.0)
                    fz = float(node.get_property("force_z") or 0.0)
                    vec = [fx, fy, fz]
                    import math as _m

                    force_mag = _m.sqrt(fx * fx + fy * fy + fz * fz)
                    force_label = f"F {self.viewer._format_force_magnitude(force_mag)}"
                    for g in faces:
                        is_mesh_selection = isinstance(g, dict) and (
                            g.get("mesh_selection") or g.get("node_ids") is not None
                        )
                        if is_mesh_selection:
                            load_faces.append({"mesh_selection": g})
                        else:
                            load_faces.append(g)
                        centroid = self._get_face_centroid(g)
                        # FIX #V2: pass magnitude for log-scaled arrow in the viewer
                        load_vectors.append(
                            {
                                "centroid": centroid,
                                "face": None if is_mesh_selection else g,
                                "mesh_selection": g if is_mesh_selection else None,
                                "vector": vec,
                                "magnitude_N": force_mag,
                                "quantity_type": "force",
                                "label": force_label,
                                "points": g.get("points")
                                if isinstance(g, dict)
                                else None,
                            }
                        )
                    if not faces and force_mag > 1e-9:
                        centroid, samples = _condition_centroid_and_points(node)
                        if centroid is not None:
                            load_vectors.append(
                                {
                                    "centroid": centroid,
                                    "points": samples,
                                    "vector": vec,
                                    "magnitude_N": force_mag,
                                    "quantity_type": "force",
                                    "label": force_label,
                                }
                            )
                except Exception:
                    logger.debug("Optional UI operation failed.", exc_info=True)

            # ---- PressureLoadNode ─ yellow face highlight, no arrow (normal varies) ----
            elif cls == "PressureLoadNode":
                if anchor_cls == "ConstraintNode":
                    continue
                faces = _get_faces_for_bc_node(node)
                try:
                    pressure_mpa = abs(float(node.get_property("pressure") or 0.0))
                    pressure_direction = str(node.get_property("direction") or "Inward")
                except Exception:
                    pressure_mpa = 0.0
                    pressure_direction = "Inward"
                for g in faces:
                    load_faces.append(
                        {
                            "face": g,
                            "load_type": "Pressure",
                            "pressure_mpa": pressure_mpa,
                            "direction": pressure_direction,
                            "color": "#ff3dbb",
                        }
                    )

            # ---- ImpactConditionNode ─ yellow face highlight + cyan velocity arrow ----
            elif cls == "ImpactConditionNode":
                if anchor_cls == "ConstraintNode":
                    continue
                faces = _get_faces_for_bc_node(node, extra_port_names={"impact_face"})
                try:
                    vx = float(node.get_property("velocity_x") or 0.0)
                    vy = float(node.get_property("velocity_y") or 0.0)
                    vz = float(node.get_property("velocity_z") or 0.0)
                except Exception:
                    vx, vy, vz = 0.0, 0.0, 0.0
                import math as _m

                v_mag = _m.sqrt(vx * vx + vy * vy + vz * vz)
                application_scope = str(
                    node.get_property("application_scope") or ""
                ).strip()
                try:
                    wall_gap = float(node.get_property("wall_gap_mm") or 0.0)
                except (TypeError, ValueError):
                    wall_gap = 0.0
                velocity_label = f"v {v_mag:.3g} m/s"
                arrow_emitted = False
                for g in faces:
                    try:
                        is_mesh_selection = isinstance(g, dict) and (
                            g.get("mesh_selection") or g.get("node_ids") is not None
                        )
                        centroid = self._get_face_centroid(g)
                        load_faces.append(
                            {
                                "face": g,
                                "load_type": "Impact",
                                "target_kind": (
                                    g.get("boundary_kind")
                                    if isinstance(g, dict)
                                    else None
                                ),
                                "centroid": centroid,
                                "vector": [vx, vy, vz],
                                "application_scope": application_scope,
                                "wall_gap_mm": wall_gap,
                            }
                        )
                        load_vectors.append(
                            {
                                "centroid": centroid,
                                "face": None if is_mesh_selection else g,
                                "mesh_selection": g if is_mesh_selection else None,
                                "vector": [vx, vy, vz],
                                "magnitude_N": v_mag,
                                "quantity_type": "velocity",
                                "label": velocity_label,
                                "color": "#00e5ff",
                            }
                        )
                        arrow_emitted = True
                    except Exception:
                        logger.debug("Optional UI operation failed.", exc_info=True)
                # Moving-body impacts have no face by design. Put the velocity
                # arrow at the displayed body's center, never at global origin.
                if not arrow_emitted and v_mag > 1e-9:
                    try:
                        bounds = self.viewer.current_actor.GetBounds()
                        body_center = [
                            0.5 * (bounds[0] + bounds[1]),
                            0.5 * (bounds[2] + bounds[3]),
                            0.5 * (bounds[4] + bounds[5]),
                        ]
                    except Exception:
                        body_center = [0.0, 0.0, 0.0]
                    load_faces.append(
                        {
                            "face": None,
                            "load_type": "Impact",
                            "centroid": body_center,
                            "vector": [vx, vy, vz],
                            "application_scope": application_scope,
                            "wall_gap_mm": wall_gap,
                        }
                    )
                    load_vectors.append(
                        {
                            "centroid": body_center,
                            "vector": [vx, vy, vz],
                            "magnitude_N": v_mag,
                            "quantity_type": "velocity",
                            "label": velocity_label,
                            "color": "#00e5ff",
                        }
                    )

            # ---- Topology study-definition nodes -------------------------
            elif cls == "TopologySupportNode":
                if (
                    anchor_cls.startswith("Topology")
                    and anchor_cls
                    != "TopologyOptVoxelNode"
                    and node is not anchor_node
                ):
                    continue
                faces = _get_faces_for_bc_node(node, extra_port_names={"target_region"})
                support_type = str(node.get_property("support_type") or "Fixed")
                fixed_dofs = {
                    "Fixed": [0, 1, 2],
                    "Roller X": [0],
                    "Roller Y": [1],
                    "Roller Z": [2],
                    "Block X Translation": [0],
                    "Block Y Translation": [1],
                    "Block Z Translation": [2],
                    "Symmetry X": [0],
                    "Symmetry Y": [1],
                    "Symmetry Z": [2],
                }.get(support_type, [0, 1, 2])
                viz = {
                    "constraint_type": support_type,
                    "color": "#00D7FF",
                    "fixed_dofs": fixed_dofs,
                }
                constraint_faces.extend(
                    {"face": geometry, "viz": viz} for geometry in faces
                )

            elif cls == "TopologyLoadNode":
                if (
                    anchor_cls.startswith("Topology")
                    and anchor_cls
                    != "TopologyOptVoxelNode"
                    and node is not anchor_node
                ):
                    continue
                faces = _get_faces_for_bc_node(node, extra_port_names={"target_region"})
                try:
                    vector = np.asarray(
                        [
                            float(node.get_property("force_x") or 0.0),
                            float(node.get_property("force_y") or 0.0),
                            float(node.get_property("force_z") or 0.0),
                        ]
                    )
                    magnitude = float(np.linalg.norm(vector))
                except Exception:
                    vector = np.zeros(3)
                    magnitude = 0.0
                topology_force_label = (
                    f"F {self.viewer._format_force_magnitude(magnitude)}"
                )
                try:
                    source_name = str(node.name() or "")
                except Exception:
                    source_name = ""
                first_token = source_name.split(" ", 1)[0].upper()
                load_case = (
                    first_token
                    if first_token.startswith("LC")
                    and first_token[2:].isdigit()
                    else None
                )
                for geometry in faces:
                    is_mesh = isinstance(geometry, dict)
                    centroid = self._get_face_centroid(geometry)
                    face_key = tuple(
                        round(float(component), 5) for component in centroid
                    )
                    if face_key not in topology_load_face_keys:
                        topology_load_face_keys.add(face_key)
                        load_faces.append(
                            {"mesh_selection": geometry} if is_mesh else geometry
                        )
                    load_vectors.append(
                        {
                            "centroid": centroid,
                            "face": None if is_mesh else geometry,
                            "mesh_selection": geometry if is_mesh else None,
                            "vector": vector.tolist(),
                            "magnitude_N": magnitude,
                            "quantity_type": "force",
                            "label": topology_force_label,
                            "load_case": load_case,
                        }
                    )

        return constraint_faces, load_faces, load_vectors

    def _show_bc_for_node(self, node):
        """
        Collect all BC data visible from *node* and push it to both the viewer
        cache and render_bc_overlays().  Safe to call after render_shape() or
        render_simulation() — overlays are layered on top.
        """
        try:
            cls = node.__class__.__name__
            if cls in ("SelectFaceNode", "InteractiveSelectFaceNode"):
                result = getattr(node, "_last_result", None)
                faces = []
                if isinstance(result, dict):
                    entity_type = str(
                        result.get("entity_type")
                        or node.get_property("entity_type")
                        or "Face"
                    ).title()
                    faces = [
                        entity
                        for entity in (
                            result.get("entities") or result.get("faces") or []
                        )
                        if entity is not None
                    ]
                    if not faces:
                        entity = result.get("entity") or result.get("face")
                        if entity is not None:
                            faces = [entity]
                if faces:
                    selections = [
                        {
                            "face": entity,
                            "load_type": "Selection",
                            "entity_type": entity_type,
                        }
                        for entity in faces
                    ]
                    self.viewer.set_bc_overlay_data(load_faces=selections)
                    self.viewer.render_bc_overlays(load_faces=selections)
                    return

            c_faces, l_faces, l_vecs = self._collect_bc_for_node(node)
            has_any = bool(c_faces or l_faces or l_vecs)
            if has_any:
                self.viewer.set_bc_overlay_data(
                    constraint_faces=c_faces or None,
                    load_faces=l_faces or None,
                    load_vectors=l_vecs or None,
                )
                self.viewer.render_bc_overlays(
                    constraint_faces=c_faces or None,
                    load_faces=l_faces or None,
                    load_vectors=l_vecs or None,
                )
            else:
                # Nothing to show — clear any stale overlays
                self.viewer.set_bc_overlay_data()
                self.viewer.render_bc_overlays()
        except Exception:
            logger.debug("Optional UI operation failed.", exc_info=True)
