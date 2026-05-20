# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""The Topology Opt (Voxel) graph node — the Qt-facing wrapper around the solver."""
from __future__ import annotations

import logging
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pylcss.design_studio.core.base_node import CadQueryNode
from .boundary_conditions import (
    VoxelBC, ManufacturingConstraints,
    _parse_support, _parse_support_region_dofs, _parse_region_boxes,
    _parse_region_cylinders, _parse_load_cases,
)
from .presets import (
    industrial_topopt_defaults,
    INDUSTRIAL_WORKFLOW_MODES, INDUSTRIAL_DESIGN_GOALS,
    INDUSTRIAL_QUALITY_PRESETS, INDUSTRIAL_MANUFACTURING_PROCESSES,
)
from .solver import TopologyOptVoxelSolver, TopologyOptVoxelProblem
from .recovery import _recover_voxel_shape

logger = logging.getLogger(__name__)

def _flatten(values: Any) -> List[Any]:
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        out: List[Any] = []
        for item in values:
            out.extend(_flatten(item))
        return out
    return [values]


def _mesh_bounds(mesh: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if mesh is None or not hasattr(mesh, 'p'):
        return None
    pts = np.asarray(mesh.p, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] == 0:
        return None
    return pts[:3].min(axis=1), pts[:3].max(axis=1)


def _bbox_tuple(bb: Any) -> Optional[Tuple[float, float, float, float, float, float]]:
    if bb is None:
        return None
    if isinstance(bb, dict):
        try:
            return (
                float(bb['xmin']), float(bb['xmax']),
                float(bb['ymin']), float(bb['ymax']),
                float(bb['zmin']), float(bb['zmax']),
            )
        except Exception:
            return None
    try:
        return (
            float(bb.xmin), float(bb.xmax),
            float(bb.ymin), float(bb.ymax),
            float(bb.zmin), float(bb.zmax),
        )
    except Exception:
        return None


def _entry_bboxes(entry: Dict[str, Any]) -> List[Tuple[float, float, float, float, float, float]]:
    bboxes: List[Tuple[float, float, float, float, float, float]] = []
    for face in entry.get('geometries') or []:
        try:
            bb = _bbox_tuple(face.BoundingBox())
        except Exception:
            bb = None
        if bb is not None:
            bboxes.append(bb)

    viz = entry.get('viz') if isinstance(entry.get('viz'), dict) else {}
    for face in viz.get('faces') or []:
        bb = _bbox_tuple(face.get('bbox') if isinstance(face, dict) else None)
        if bb is not None:
            bboxes.append(bb)
    bb = _bbox_tuple(viz.get('bbox'))
    if bb is not None:
        bboxes.append(bb)
    return bboxes


def _fraction(value: float, lo: float, hi: float, *, invert: bool = False) -> float:
    span = max(float(hi) - float(lo), 1e-12)
    frac = (float(value) - float(lo)) / span
    if invert:
        frac = 1.0 - frac
    return float(np.clip(frac, 0.0, 1.0))


def _fraction_box(
    bbox: Tuple[float, float, float, float, float, float],
    bounds: Tuple[np.ndarray, np.ndarray],
    pad: float = 0.02,
) -> Tuple[float, float, float, float, float, float]:
    mins, maxs = bounds
    vals = [
        _fraction(bbox[0], mins[0], maxs[0]),
        _fraction(bbox[1], mins[0], maxs[0]),
        _fraction(bbox[2], mins[1], maxs[1]),
        _fraction(bbox[3], mins[1], maxs[1]),
        _fraction(bbox[4], mins[2], maxs[2]),
        _fraction(bbox[5], mins[2], maxs[2]),
    ]
    for i in (0, 2, 4):
        if abs(vals[i + 1] - vals[i]) < pad:
            center = 0.5 * (vals[i] + vals[i + 1])
            vals[i] = max(0.0, center - pad)
            vals[i + 1] = min(1.0, center + pad)
    return tuple(vals)  # type: ignore[return-value]


def _fraction_center(
    bbox: Tuple[float, float, float, float, float, float],
    bounds: Tuple[np.ndarray, np.ndarray],
) -> Tuple[float, float, float]:
    mins, maxs = bounds
    return (
        _fraction(0.5 * (bbox[0] + bbox[1]), mins[0], maxs[0]),
        _fraction(0.5 * (bbox[2] + bbox[3]), mins[1], maxs[1]),
        _fraction(0.5 * (bbox[4] + bbox[5]), mins[2], maxs[2]),
    )


def _is_load_payload(entry: Any) -> bool:
    return (
        isinstance(entry, dict)
        and str(entry.get('type') or '').lower() in {'force', 'pressure', 'gravity'}
    )


def _point_xyz(point: Any) -> Optional[np.ndarray]:
    try:
        return np.asarray([float(point.x), float(point.y), float(point.z)])
    except Exception:
        pass
    try:
        values = list(point)
        if len(values) >= 3:
            return np.asarray([float(values[0]), float(values[1]), float(values[2])])
    except Exception:
        pass
    return None


def _iter_load_faces(load: Dict[str, Any]) -> List[Any]:
    faces = list(load.get('geometries') or [])
    if not faces and load.get('geometry') is not None:
        faces = [load.get('geometry')]
    return [face for face in faces if face is not None]


def _bbox_pressure_fallback(
    bbox: Tuple[float, float, float, float, float, float],
    bounds: Tuple[np.ndarray, np.ndarray],
    pressure: float,
) -> Tuple[float, float, float]:
    """Approximate a planar pressure face as one resultant force."""
    mins, maxs = bounds
    extents = np.asarray([
        max(0.0, bbox[1] - bbox[0]),
        max(0.0, bbox[3] - bbox[2]),
        max(0.0, bbox[5] - bbox[4]),
    ], dtype=float)
    axis = int(np.argmin(extents))
    area_axes = [idx for idx in range(3) if idx != axis]
    area = float(np.prod(extents[area_axes]))
    center = np.asarray([
        0.5 * (bbox[0] + bbox[1]),
        0.5 * (bbox[2] + bbox[3]),
        0.5 * (bbox[4] + bbox[5]),
    ], dtype=float)
    midpoint = 0.5 * (mins[:3] + maxs[:3])
    sign = 1.0 if center[axis] >= midpoint[axis] else -1.0
    force = np.zeros(3, dtype=float)
    force[axis] = float(pressure) * area * sign
    return float(force[0]), float(force[1]), float(force[2])


def _pressure_face_patches(
    face: Any,
    bounds: Tuple[np.ndarray, np.ndarray],
    pressure: float,
    *,
    max_bins: Tuple[int, int, int] = (12, 12, 4),
) -> List[Tuple[Tuple[float, float, float, float, float, float], float, float, float]]:
    """Approximate pressure on flat or curved CAD faces as voxel patch loads.

    The topology solver accepts nodal box forces, not native pressure BCs.
    Tessellating the face lets curved pressure, such as pressure on a hole
    wall, remain self-equilibrated instead of collapsing into one misleading
    resultant vector.
    """
    if abs(float(pressure)) <= 1e-30:
        return []

    try:
        vertices, triangles = face.tessellate(0.75)
    except TypeError:
        try:
            vertices, triangles = face.tessellate(1.0)
        except Exception:
            return []
    except Exception:
        return []

    points = [_point_xyz(v) for v in vertices]
    if not points or any(p is None for p in points):
        return []
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return []

    mins, maxs = bounds
    span = np.maximum(maxs[:3] - mins[:3], 1e-12)
    bin_counts = np.maximum(np.asarray(max_bins, dtype=int), 1)
    bins: Dict[Tuple[int, int, int], List[Any]] = {}

    for tri in triangles:
        try:
            ids = [int(i) for i in list(tri)[:3]]
        except Exception:
            continue
        if len(ids) != 3 or any(i < 0 or i >= len(pts) for i in ids):
            continue

        tri_pts = pts[ids, :3]
        area_vec = 0.5 * np.cross(tri_pts[1] - tri_pts[0], tri_pts[2] - tri_pts[0])
        if float(np.linalg.norm(area_vec)) <= 1e-12:
            continue

        force = float(pressure) * area_vec
        center = np.mean(tri_pts, axis=0)
        frac = np.clip((center - mins[:3]) / span, 0.0, 1.0)
        key_arr = np.floor(frac * bin_counts).astype(int)
        key_arr = np.clip(key_arr, 0, bin_counts - 1)
        key = (int(key_arr[0]), int(key_arr[1]), int(key_arr[2]))

        lo = np.min(tri_pts, axis=0)
        hi = np.max(tri_pts, axis=0)
        bbox = [lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]]
        if key not in bins:
            bins[key] = [bbox, force.astype(float)]
        else:
            acc_bbox, acc_force = bins[key]
            acc_bbox[0] = min(acc_bbox[0], bbox[0])
            acc_bbox[1] = max(acc_bbox[1], bbox[1])
            acc_bbox[2] = min(acc_bbox[2], bbox[2])
            acc_bbox[3] = max(acc_bbox[3], bbox[3])
            acc_bbox[4] = min(acc_bbox[4], bbox[4])
            acc_bbox[5] = max(acc_bbox[5], bbox[5])
            acc_force += force

    patches: List[Tuple[Tuple[float, float, float, float, float, float], float, float, float]] = []
    for _, (bbox, force) in sorted(bins.items(), key=lambda item: item[0]):
        if float(np.linalg.norm(force)) <= 1e-12:
            continue
        frac_box = _fraction_box(tuple(float(v) for v in bbox), bounds, pad=0.012)
        patches.append((frac_box, float(force[0]), float(force[1]), float(force[2])))
    return patches


def _pressure_load_patches(
    load: Dict[str, Any],
    bounds: Tuple[np.ndarray, np.ndarray],
) -> List[Tuple[Tuple[float, float, float, float, float, float], float, float, float]]:
    try:
        pressure = float(load.get('pressure', load.get('magnitude', 0.0)))
    except Exception:
        return []
    patches: List[Tuple[Tuple[float, float, float, float, float, float], float, float, float]] = []
    for face in _iter_load_faces(load):
        patches.extend(_pressure_face_patches(face, bounds, pressure))
    if patches:
        return patches

    for bbox in _entry_bboxes(load):
        frac_box = _fraction_box(bbox, bounds)
        fx, fy, fz = _bbox_pressure_fallback(bbox, bounds, pressure)
        if (fx * fx + fy * fy + fz * fz) > 1e-24:
            patches.append((frac_box, fx, fy, fz))
    return patches


def _bounds_payload(bounds: Optional[Tuple[np.ndarray, np.ndarray]]) -> Optional[Dict[str, List[float]]]:
    if bounds is None:
        return None
    mins, maxs = bounds
    return {
        'min': [float(v) for v in mins[:3]],
        'max': [float(v) for v in maxs[:3]],
    }


# ---------------------------------------------------------------------------
# CadQueryNode
# ---------------------------------------------------------------------------

class TopologyOptVoxelNode(CadQueryNode):
    """3-D SIMP topology optimisation using pyMOTO on a structured voxel grid.

    The graph-facing inputs match the normal FEM solver flow: mesh, material,
    constraints, and loads. Solver resolution and optimiser controls live as
    node properties instead of cluttering the graph with hyperparameter ports.
    """
    __identifier__ = 'com.cad.sim.topopt_voxel'
    NODE_NAME = 'Topology Opt (3D Voxel)'

    def __init__(self):
        super().__init__()
        self.add_output('result', color=(180, 255, 180))
        self.add_output('recovered_shape', color=(100, 255, 100))
        self.add_input('mesh',        color=(200, 100, 200))
        self.add_input('material',    color=(200, 200, 200))
        self.add_input('constraints', color=(255, 100, 100), multi_input=True)
        self.add_input('loads',       color=(255, 255,   0), multi_input=True)
        # Guided workflow
        self.create_property('workflow_mode',
                             'Guided',
                             widget_type='combo',
                             items=list(INDUSTRIAL_WORKFLOW_MODES))
        self.create_property('design_goal',
                             'Lightweight Stiffness',
                             widget_type='combo',
                             items=list(INDUSTRIAL_DESIGN_GOALS))
        self.create_property('quality_preset',
                             'Balanced',
                             widget_type='combo',
                             items=list(INDUSTRIAL_QUALITY_PRESETS))
        self.create_property('manufacturing_process',
                             'None',
                             widget_type='combo',
                             items=list(INDUSTRIAL_MANUFACTURING_PROCESSES))
        self.create_property('advanced_settings_visible', False, widget_type='bool')
        self.create_property('validate_after_optimize', False, widget_type='bool')
        self.create_property('validation_quality',
                             'Standard',
                             widget_type='combo',
                             items=['Standard', 'Mesh Convergence'])
        self.create_property('generate_cad_after_optimize', False, widget_type='bool')
        self.create_property('cad_reconstruction_method',
                             'Recovered Shape',
                             widget_type='combo',
                             items=['Recovered Shape'])
        self.create_property('cad_export_filename',
                             'topology_optimized.step',
                             widget_type='string')
        # ── Domain ────────────────────────────────────────────────────────
        self.create_property('nelx',    30,    widget_type='int')
        self.create_property('nely',    20,    widget_type='int')
        self.create_property('nelz',    10,    widget_type='int')
        self.create_property('volfrac', 0.5,   widget_type='float')
        self.create_property('rmin',    1.5,   widget_type='float')
        self.create_property('penal',   3.0,   widget_type='float')
        self.create_property('density_cutoff', 0.45, widget_type='float')
        self.create_property('visualization', 'Density', widget_type='combo',
                             items=['Density', 'Recovered Shape'])
        # ── STL post-processing ───────────────────────────────────────────
        # When True, applies the trimesh print-ready pipeline (hole fill +
        # light Humphrey smoothing + optional decimation) on top of marching cubes.
        self.create_property('print_ready_mesh', False, widget_type='bool')
        # 1.0 = no decimation; lower values reduce face count (requires
        # `fast_simplification` to be installed — silently skipped otherwise).
        self.create_property('mesh_decimate_ratio', 1.0, widget_type='float')
        self.create_property('E0',      1.0,   widget_type='float')
        self.create_property('Emin',    1e-9,  widget_type='float')
        self.create_property('nu',      0.3,   widget_type='float')

        # ── Solver ────────────────────────────────────────────────────────
        self.create_property('optimizer', 'OC', widget_type='combo',
                             items=['OC', 'MMA'])
        self.create_property('max_iter', 80,   widget_type='int')
        # `tol` is the relative-compliance-change convergence threshold; the run
        # stops once it holds for `convergence_patience` consecutive iterations.
        self.create_property('tol',      0.01, widget_type='float')
        self.create_property('convergence_patience', 5, widget_type='int')

        # ── Phase 3: stress constraint ────────────────────────────────────
        # When enabled, adds a P-norm aggregated von Mises constraint and
        # forces MMA (OC cannot handle multiple constraints).
        self.create_property('stress_constraint', False, widget_type='bool')
        self.create_property('yield_stress',      1.0,   widget_type='float')
        # SIMP stress relaxation exponent.  1.0 = linear (industry default);
        # higher (1.5–2.0) more strictly excludes void elements.
        self.create_property('stress_penalty',    1.0,   widget_type='float')
        # PNorm exponent.  Higher → tighter approximation of max but harder
        # convergence.  8–16 typical; default 8.
        self.create_property('stress_pnorm_p',    8.0,   widget_type='float')

        # ── BC Preset (convenience only) ─────────────────────────────────
        self.create_property('bc_preset', 'Cantilever Tip', widget_type='combo',
                             items=['Custom',
                                    'Cantilever Tip',
                                    'Cantilever Distributed',
                                    'MBB Beam',
                                    'Bridge',
                                    'Cantilever Bi-Axial',
                                    'Printable Cantilever'])

        # ── Manufacturing constraints (Phase 2) ───────────────────────────
        # `symmetry`: subset of {X,Y,Z} meaning "axes to mirror across the
        # domain centre".  E.g. 'Y' = mirror plane XZ (left-right symmetric);
        # 'XY' = mirror both X and Y.
        self.create_property('symmetry',
                             'None',
                             widget_type='combo',
                             items=['None', 'X', 'Y', 'Z', 'XY', 'XZ', 'YZ', 'XYZ'])
        self.create_property('extrusion',
                             'None',
                             widget_type='combo',
                             items=['None', 'X', 'Y', 'Z'])
        # AM build direction.  '+Y' means parts grow upward in +Y; voxels are
        # only allowed if supported by a 3×3 neighbourhood of denser voxels
        # in the layer below (45° self-supporting rule).
        self.create_property('overhang_build_axis',
                             'None',
                             widget_type='combo',
                             items=['None', '+X', '-X', '+Y', '-Y', '+Z', '-Z'])
        # Max member size — radius in voxels (0 = disabled); local-mean
        # threshold defaults to 0.6 and isn't user-exposed (industry default).
        self.create_property('max_member_size_voxels', 0.0, widget_type='float')
        # N-fold rotational pattern. 1 = disabled.
        self.create_property('pattern_repeat', 1, widget_type='int')
        self.create_property('pattern_axis',
                             'Y',
                             widget_type='combo',
                             items=['X', 'Y', 'Z'])

        # ── Face supports ─────────────────────────────────────────────────
        _support_items = ['None', 'Fix X', 'Fix Y', 'Fix Z',
                          'Fix XY', 'Fix YZ', 'Fix XZ', 'Fix XYZ']
        self.create_property('left_support',   'Fix XYZ', widget_type='combo', items=_support_items)
        self.create_property('right_support',  'None',    widget_type='combo', items=_support_items)
        self.create_property('top_support',    'None',    widget_type='combo', items=_support_items)
        self.create_property('bottom_support', 'None',    widget_type='combo', items=_support_items)
        self.create_property('front_support',  'None',    widget_type='combo', items=_support_items)
        self.create_property('back_support',   'None',    widget_type='combo', items=_support_items)
        self.create_property('support_regions', '[]', widget_type='text')

        # ── Non-design (passive) regions ──────────────────────────────────
        # JSON list of {"x":[a,b],"y":[a,b],"z":[a,b]} in fractional coords.
        # Voxels inside solid_regions are clamped to ρ=1, void_regions to ρ≈0.
        self.create_property('solid_regions', '[]', widget_type='text')
        self.create_property('void_regions',  '[]', widget_type='text')

        # ── Multi-load case JSON ──────────────────────────────────────────
        # When non-empty, overrides the single-force properties below.
        # Format:
        #   [{"name":"LC1","weight":1.0,
        #     "point_forces":[{"x":1.0,"y":0.5,"z":0.5,"fx":0,"fy":-1,"fz":0}],
        #     "distributed_forces":[{"face":"right","fx":0,"fy":-1,"fz":0}]}]
        self.create_property('load_cases', '[]', widget_type='text')

        # ── Force ─────────────────────────────────────────────────────────
        self.create_property('force_type', 'Point', widget_type='combo',
                             items=['Point', 'Distributed Face'])
        self.create_property('force_face', 'Right', widget_type='combo',
                             items=['Left', 'Right', 'Top', 'Bottom', 'Front', 'Back'])
        # Point force location as fractions of domain size [0, 1]
        self.create_property('force_ix_frac',   1.0,  widget_type='float')
        self.create_property('force_iy_frac',   0.5,  widget_type='float')
        self.create_property('force_iz_frac',   0.5,  widget_type='float')
        self.create_property('force_dir_x',     0.0,  widget_type='float')
        self.create_property('force_dir_y',    -1.0,  widget_type='float')
        self.create_property('force_dir_z',     0.0,  widget_type='float')
        self.create_property('force_magnitude', 1.0,  widget_type='float')

    # ── Property helpers ───────────────────────────────────────────────────

    def _apply_preset(self, preset: str) -> None:
        """Pre-fill support and force properties from a named preset."""
        # Every preset clears the manufacturing-constraint properties by
        # default; the "Printable Cantilever" preset sets them explicitly.
        _mc_defaults = dict(
            symmetry='None',
            extrusion='None',
            overhang_build_axis='None',
            max_member_size_voxels=0.0,
            pattern_repeat=1,
            pattern_axis='Y',
        )
        _presets = {
            'Cantilever Tip': dict(
                left_support='Fix XYZ', right_support='None',
                top_support='None',     bottom_support='None',
                front_support='None',   back_support='None',
                support_regions='[]',
                load_cases='[]',
                force_type='Point',
                force_ix_frac=1.0, force_iy_frac=0.5, force_iz_frac=0.5,
                force_dir_x=0.0,   force_dir_y=-1.0,  force_dir_z=0.0,
                force_magnitude=1.0,
                **_mc_defaults,
            ),
            'Cantilever Distributed': dict(
                left_support='Fix XYZ', right_support='None',
                top_support='None',     bottom_support='None',
                front_support='None',   back_support='None',
                support_regions='[]',
                load_cases='[]',
                force_type='Distributed Face', force_face='Right',
                force_dir_x=0.0,   force_dir_y=-1.0,  force_dir_z=0.0,
                force_magnitude=1.0,
                **_mc_defaults,
            ),
            'MBB Beam': dict(
                left_support='None',  right_support='None',
                top_support='None',   bottom_support='None',
                front_support='None', back_support='None',
                support_regions=json.dumps([
                    {"x": [0.00, 0.04], "y": [0.00, 0.04], "z": [0.0, 1.0], "dofs": "Fix XYZ"},
                    {"x": [0.96, 1.00], "y": [0.00, 0.04], "z": [0.0, 1.0], "dofs": "Fix YZ"},
                ]),
                load_cases='[]',
                force_type='Point',
                force_ix_frac=0.5, force_iy_frac=1.0, force_iz_frac=0.5,
                force_dir_x=0.0,   force_dir_y=-1.0,  force_dir_z=0.0,
                force_magnitude=1.0,
                **_mc_defaults,
            ),
            'Bridge': dict(
                left_support='None',  right_support='None',
                top_support='None',   bottom_support='Fix Y',
                front_support='None', back_support='None',
                support_regions='[]',
                load_cases='[]',
                force_type='Point',
                force_ix_frac=0.5, force_iy_frac=1.0, force_iz_frac=0.5,
                force_dir_x=0.0,   force_dir_y=-1.0,  force_dir_z=0.0,
                force_magnitude=1.0,
                **_mc_defaults,
            ),
            'Cantilever Bi-Axial': dict(
                left_support='Fix XYZ', right_support='None',
                top_support='None',     bottom_support='None',
                front_support='None',   back_support='None',
                support_regions='[]',
                # Two equally-weighted independent loads at the tip:
                #   downward Y force and lateral Z force.  Together they push
                #   the optimiser toward a structure that resists both — the
                #   classic demo of why multi-load matters in industry.
                load_cases=json.dumps([
                    {"name": "Tip-Down",
                     "weight": 1.0,
                     "point_forces": [
                         {"x": 1.0, "y": 0.5, "z": 0.5, "fx": 0.0, "fy": -1.0, "fz": 0.0}
                     ]},
                    {"name": "Tip-Lateral",
                     "weight": 1.0,
                     "point_forces": [
                         {"x": 1.0, "y": 0.5, "z": 0.5, "fx": 0.0, "fy": 0.0, "fz": 1.0}
                     ]},
                ]),
                **_mc_defaults,
            ),
            # AM-aware demo: tip-loaded cantilever printed in +Y, mirror in Z.
            'Printable Cantilever': dict(
                left_support='Fix XYZ', right_support='None',
                top_support='None',     bottom_support='None',
                front_support='None',   back_support='None',
                support_regions='[]',
                load_cases='[]',
                force_type='Point',
                force_ix_frac=1.0, force_iy_frac=0.5, force_iz_frac=0.5,
                force_dir_x=0.0,   force_dir_y=-1.0,  force_dir_z=0.0,
                force_magnitude=1.0,
                symmetry='Z',
                extrusion='None',
                overhang_build_axis='+Y',
            ),
        }
        if preset in _presets:
            for k, v in _presets[preset].items():
                self.set_property(k, v)

    def apply_guided_defaults(self) -> Dict[str, Any]:
        """Apply guided workflow defaults to this node and return the changes."""
        settings = industrial_topopt_defaults(
            self.get_property('design_goal'),
            self.get_property('quality_preset'),
            self.get_property('manufacturing_process'),
            nelx=self.get_property('nelx') or 30,
            nely=self.get_property('nely') or 20,
            nelz=self.get_property('nelz') or 10,
        )
        for key, value in settings.items():
            self.set_property(key, value)
        return settings

    def _run_embedded_validation(
        self,
        topo_output: Dict[str, Any],
        material: Dict[str, Any],
        constraint_list: List[Any],
        load_list: List[Any],
    ) -> Optional[Dict[str, Any]]:
        """Run the validation stage as an internal TopOpt study action."""
        from .validation import run_topopt_validation

        quality = str(self.get_property('validation_quality') or 'Standard')
        return run_topopt_validation(
            topo_output,
            material,
            constraint_list,
            load_list,
            convergence_levels=2 if quality == 'Mesh Convergence' else 1,
            max_validation_elements=500000,
            run_external_solver=True,
            deck_only=False,
            analysis_type='Linear',
            visualization='Von Mises Stress',
            deformation_scale='Auto',
        )

    def _run_embedded_cad_reconstruction(
        self,
        topo_output: Dict[str, Any],
    ) -> Any:
        """Build a CAD body as an internal TopOpt study action."""
        from .cad_reconstruction import reconstruct_topopt_cad

        self.set_property('cad_reconstruction_method', 'Recovered Shape')
        topo_payload = dict(topo_output or {})
        if 'passive_regions' not in topo_payload:
            bc = self._build_bc()
            topo_payload['passive_regions'] = {
                'solid_boxes': list(bc.solid_boxes),
                'void_boxes': list(bc.void_boxes),
                'solid_cylinders': list(bc.solid_cylinders),
                'void_cylinders': list(bc.void_cylinders),
            }
        topo_payload.setdefault(
            'extrusion_axis',
            str(self.get_property('extrusion') or 'none').strip().lower(),
        )
        return reconstruct_topopt_cad(
            topo_payload,
            source_geometry='Smooth Recovered Shape',
            max_smooth_faces=3000,
            sew_tolerance=1e-4,
            merge_angle_deg=0.0,
            density_cutoff=0.0,
        )

    def _build_bc(self) -> VoxelBC:
        """Construct a VoxelBC from the current node properties."""
        bc = VoxelBC(
            fixed_left_face_dofs   = _parse_support(self.get_property('left_support')),
            fixed_right_face_dofs  = _parse_support(self.get_property('right_support')),
            fixed_top_face_dofs    = _parse_support(self.get_property('top_support')),
            fixed_bottom_face_dofs = _parse_support(self.get_property('bottom_support')),
            fixed_front_face_dofs  = _parse_support(self.get_property('front_support')),
            fixed_back_face_dofs   = _parse_support(self.get_property('back_support')),
        )

        regions_text = str(self.get_property('support_regions') or '').strip()
        if regions_text:
            try:
                regions = json.loads(regions_text)
                if not isinstance(regions, list):
                    raise ValueError("support_regions must be a JSON list")
                for region in regions:
                    if not isinstance(region, dict):
                        continue
                    x0, x1 = region.get('x', [0.0, 0.0])
                    y0, y1 = region.get('y', [0.0, 0.0])
                    z0, z1 = region.get('z', [0.0, 1.0])
                    dofs = _parse_support_region_dofs(region.get('dofs', ''))
                    bc.fixed_boxes.append((
                        float(x0), float(x1),
                        float(y0), float(y1),
                        float(z0), float(z1),
                        dofs,
                    ))
            except Exception as exc:
                raise ValueError(f"Invalid support_regions JSON: {exc}") from exc

        # ── non-design regions ────────────────────────────────────────────
        bc.solid_boxes = _parse_region_boxes(
            self.get_property('solid_regions'), 'solid_regions'
        )
        bc.solid_cylinders = _parse_region_cylinders(
            self.get_property('solid_regions'), 'solid_regions'
        )
        bc.void_boxes = _parse_region_boxes(
            self.get_property('void_regions'),  'void_regions'
        )
        bc.void_cylinders = _parse_region_cylinders(
            self.get_property('void_regions'),  'void_regions'
        )

        # ── multi-load case (overrides legacy single-force if present) ────
        bc.load_cases = _parse_load_cases(self.get_property('load_cases'))
        if bc.load_cases:
            return bc

        # ── legacy single-force fallback ──────────────────────────────────
        mag  = float(self.get_property('force_magnitude') or 1.0)
        fdx  = float(self.get_property('force_dir_x') or 0.0)
        fdy  = float(self.get_property('force_dir_y') or -1.0)
        fdz  = float(self.get_property('force_dir_z') or 0.0)

        norm = (fdx ** 2 + fdy ** 2 + fdz ** 2) ** 0.5 or 1.0
        fdx, fdy, fdz = (fdx / norm) * mag, (fdy / norm) * mag, (fdz / norm) * mag

        ftype = self.get_property('force_type')
        if ftype == 'Point':
            ix_f = float(self.get_property('force_ix_frac') or 1.0)
            iy_f = float(self.get_property('force_iy_frac') or 0.5)
            iz_f = float(self.get_property('force_iz_frac') or 0.5)
            bc.point_forces.append((ix_f, iy_f, iz_f, fdx, fdy, fdz))
        else:  # Distributed Face
            face = (self.get_property('force_face') or 'Right').lower()
            bc.distributed_forces.append((face, fdx, fdy, fdz))

        return bc

    def _merge_graph_bcs(
        self,
        bc: VoxelBC,
        mesh: Any,
        constraints: List[Any],
        loads: List[Any],
    ) -> None:
        """Add ordinary PyLCSS face constraints and loads to the voxel BCs."""
        bounds = _mesh_bounds(mesh)
        if bounds is None:
            if constraints or loads:
                logger.warning(
                    "TopologyOptVoxelNode: cannot map graph BCs without mesh bounds."
                )
            return

        for constraint in constraints:
            if not isinstance(constraint, dict):
                continue
            dofs = constraint.get('fixed_dofs') or []
            try:
                dofs = [int(d) for d in dofs if int(d) in (0, 1, 2)]
            except Exception:
                dofs = []
            if not dofs:
                continue
            for bbox in _entry_bboxes(constraint):
                frac_box = _fraction_box(bbox, bounds)
                bc.fixed_boxes.append((*frac_box, dofs))

        for load in loads:
            if not isinstance(load, dict):
                continue
            load_type = str(load.get('type') or '').lower()
            if load_type == 'force':
                vector = load.get('vector')
                try:
                    fx, fy, fz = (float(vector[0]), float(vector[1]), float(vector[2]))
                except Exception:
                    continue
                bboxes = _entry_bboxes(load)
                if not bboxes:
                    continue
                scale = 1.0 / max(1, len(bboxes))
                for bbox in bboxes:
                    frac_box = _fraction_box(bbox, bounds)
                    # Graph-selected load faces become distributed patch loads,
                    # not mathematical point loads.  The same patch is also kept
                    # local so CAD face loads stay tied to their source face.
                    bc.box_forces.append((*frac_box, fx * scale, fy * scale, fz * scale))
            elif load_type == 'pressure':
                pressure_patches = _pressure_load_patches(load, bounds)
                if not pressure_patches:
                    logger.warning(
                        "TopologyOptVoxelNode: pressure load could not be "
                        "mapped to voxel patch forces."
                    )
                    continue
                for frac_box, fx, fy, fz in pressure_patches:
                    bc.box_forces.append((*frac_box, fx, fy, fz))
            else:
                logger.debug(
                    "TopologyOptVoxelNode: load type %r is not supported by "
                    "the voxel optimiser.",
                    load_type,
                )

    # ── Node run ───────────────────────────────────────────────────────────

    def run(self, progress_callback=None) -> Optional[Dict[str, Any]]:
        preset = self.get_property('bc_preset')
        if preset != 'Custom':
            self._apply_preset(preset)

        mesh = self.get_input_value('mesh', None)
        bounds = _mesh_bounds(mesh)
        material = self.get_input_value('material', None)
        material = material if isinstance(material, dict) else {}
        raw_constraints = _flatten(self.get_input_list('constraints'))
        constraint_list: List[Any] = []
        load_like_constraints: List[Any] = []
        for item in raw_constraints:
            if _is_load_payload(item):
                load_like_constraints.append(item)
            else:
                constraint_list.append(item)
        if load_like_constraints:
            logger.warning(
                "TopologyOptVoxelNode: %d load payload(s) were connected to "
                "the constraints input; treating them as loads.",
                len(load_like_constraints),
            )
        load_list = _flatten(self.get_input_list('loads')) + load_like_constraints
        bc = self._build_bc()
        if constraint_list:
            bc.fixed_left_face_dofs = []
            bc.fixed_right_face_dofs = []
            bc.fixed_top_face_dofs = []
            bc.fixed_bottom_face_dofs = []
            bc.fixed_front_face_dofs = []
            bc.fixed_back_face_dofs = []
            bc.fixed_boxes = []
        if load_list:
            bc.point_forces = []
            bc.box_forces = []
            bc.distributed_forces = []
            bc.load_cases = []  # graph-supplied loads override node-property load cases
        self._merge_graph_bcs(bc, mesh, constraint_list, load_list)

        nelx = int(self.get_property('nelx') or 30)
        nely = int(self.get_property('nely') or 20)
        nelz = int(self.get_property('nelz') or 10)
        unitx = unity = unitz = 1.0
        if bounds is not None:
            mins, maxs = bounds
            span = np.maximum(maxs[:3] - mins[:3], 1e-12)
            unitx = float(span[0] / max(nelx, 1))
            unity = float(span[1] / max(nely, 1))
            unitz = float(span[2] / max(nelz, 1))

        mc = ManufacturingConstraints(
            symmetry            = (str(self.get_property('symmetry') or 'None')).lower(),
            extrusion           = (str(self.get_property('extrusion') or 'None')).lower(),
            overhang_build_axis = (str(self.get_property('overhang_build_axis') or 'None')).lower(),
            max_member_size_voxels = float(self.get_property('max_member_size_voxels') or 0.0),
            pattern_repeat = int(self.get_property('pattern_repeat') or 1),
            pattern_axis   = (str(self.get_property('pattern_axis') or 'Y')).lower(),
        )

        problem = TopologyOptVoxelProblem(
            nelx     = nelx,
            nely     = nely,
            nelz     = nelz,
            E0       = float(material.get('E', self.get_property('E0') or 1.0)),
            Emin     = float(self.get_property('Emin')  or 1e-9),
            nu       = float(material.get('nu', self.get_property('nu') or 0.3)),
            penal    = float(self.get_property('penal') or 3.0),
            volfrac  = float(self.get_property('volfrac') or 0.5),
            rmin     = float(self.get_property('rmin') or 1.5),
            unitx    = unitx,
            unity    = unity,
            unitz    = unitz,
            optimizer = self.get_property('optimizer') or 'OC',
            max_iter = int(self.get_property('max_iter') or 80),
            tol      = float(self.get_property('tol')   or 0.01),
            patience = int(self.get_property('convergence_patience') or 5),
            bc       = bc,
            mc       = mc,
            stress_constraint_enabled = bool(self.get_property('stress_constraint')),
            yield_stress              = float(self.get_property('yield_stress') or 1.0),
            stress_penalty            = float(self.get_property('stress_penalty') or 1.0),
            stress_pnorm_p            = float(self.get_property('stress_pnorm_p') or 8.0),
        )

        solver = TopologyOptVoxelSolver(problem)

        def _cb(it: int, comp: float, change: float, density: np.ndarray) -> None:
            if progress_callback is not None:
                try:
                    progress_callback(
                        {
                            'type': 'topopt_voxel',
                            'density': density,
                            'grid_shape': density.shape,
                            'bounds': _bounds_payload(bounds),
                            'density_cutoff': float(self.get_property('density_cutoff') or 0.45),
                            '_preview': True,
                        },
                        density,
                        max(0, it - 1),
                        problem.max_iter,
                    )
                except Exception:
                    pass

        try:
            result = solver.run(callback=_cb)
        except Exception as exc:
            logger.exception("TopologyOptVoxelNode: solver error")
            self.set_error(str(exc))
            return None

        logger.info("TopologyOptVoxelNode: %s", result.message)
        density = np.asarray(result.density, dtype=float)
        density_cutoff = float(self.get_property('density_cutoff') or 0.45)
        print_ready = bool(self.get_property('print_ready_mesh'))
        decimate    = float(self.get_property('mesh_decimate_ratio') or 1.0)
        recovered = _recover_voxel_shape(
            density, bounds, density_cutoff,
            print_ready=print_ready,
            decimate_ratio=decimate,
            solid_boxes=bc.solid_boxes,
            void_boxes=bc.void_boxes,
            solid_cylinders=bc.solid_cylinders,
            void_cylinders=bc.void_cylinders,
            extrusion_axis=mc.extrusion,
        )
        if bounds is not None:
            mins, maxs = bounds
            total_volume = float(np.prod(np.maximum(maxs[:3] - mins[:3], 0.0)))
        else:
            total_volume = float(np.prod(density.shape))
        final_volume = float(np.mean(density) * total_volume)
        material_density = float(material.get('rho', material.get('density', 0.0)))
        output: Dict[str, Any] = {
            'type': 'topopt_voxel',
            'density': density,
            'design_density': (
                np.asarray(result.design_density, dtype=float)
                if result.design_density is not None else None
            ),
            'grid_shape': density.shape,
            'bounds': _bounds_payload(bounds),
            'density_cutoff': density_cutoff,
            'recovered_shape': recovered,
            'extrusion_axis': mc.extrusion,
            'visualization_mode': self.get_property('visualization') or 'Density',
            'target_vol_frac': problem.volfrac,
            'final_vol_frac': float(np.mean(density)),
            'volume': final_volume,
            'total_volume': total_volume,
            'mass': final_volume * material_density,
            'compliance': (
                float(result.compliance_history[-1])
                if result.compliance_history else None
            ),
            'stress_pnorm': (
                float(result.stress_history[-1])
                if result.stress_history else None
            ),
            'iterations': result.n_iter,
            'converged': result.converged,
            'message': result.message,
            'compliance_history': result.compliance_history,
            'change_history': result.change_history,
            'stress_history': result.stress_history,
            'passive_regions': {
                'solid_boxes': list(bc.solid_boxes),
                'void_boxes': list(bc.void_boxes),
                'solid_cylinders': list(bc.solid_cylinders),
                'void_cylinders': list(bc.void_cylinders),
            },
        }

        warnings_out: List[str] = []
        if bool(self.get_property('validate_after_optimize')):
            try:
                validation = self._run_embedded_validation(
                    output, material, constraint_list, load_list,
                )
                if validation is not None:
                    output['validation'] = validation
                    study = validation.get('convergence_study') if isinstance(validation, dict) else None
                    output['validation_summary'] = {
                        'max_stress': validation.get('max_stress_gauss'),
                        'compliance': validation.get('compliance'),
                        'converged': (
                            study.get('converged')
                            if isinstance(study, dict) else None
                        ),
                    }
            except Exception as exc:
                msg = f"Validation skipped/failed: {exc}"
                logger.warning("TopologyOptVoxelNode: %s", msg)
                output['validation_error'] = str(exc)
                warnings_out.append(msg)

        if bool(self.get_property('generate_cad_after_optimize')):
            try:
                cad_shape = self._run_embedded_cad_reconstruction(output)
                output['cad_shape'] = cad_shape
                output['shape'] = cad_shape
                try:
                    solid = cad_shape.val() if hasattr(cad_shape, 'val') else cad_shape
                    output['cad_reconstruction'] = {
                        'method': 'Recovered Shape',
                        'valid': bool(solid.isValid()) if hasattr(solid, 'isValid') else None,
                        'volume': float(solid.Volume()) if hasattr(solid, 'Volume') else None,
                    }
                except Exception:
                    output['cad_reconstruction'] = {
                        'method': 'Recovered Shape',
                    }
            except Exception as exc:
                msg = f"CAD reconstruction skipped/failed: {exc}"
                logger.warning("TopologyOptVoxelNode: %s", msg)
                output['cad_error'] = str(exc)
                warnings_out.append(msg)

        if warnings_out:
            output['warnings'] = warnings_out
        return output
