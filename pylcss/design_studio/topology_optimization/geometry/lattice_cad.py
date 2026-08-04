# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Analytic CAD for strut lattices, built from centrelines instead of voxels.

Why this exists
---------------
The general topology result has no closed-form surface, so it earns the long
road: supersample the density, extract an isosurface, repair the mesh, and sew
one planar B-rep face per triangle. That is the right pipeline for an organic
optimized blob.

It is the wrong pipeline for a strut lattice, and expensively so. A cubic or
octet lattice on a recovery grid is a million-triangle isosurface of a shape we
already know exactly: :class:`~..manufacturing.member_sizing.OptimizedMemberPlan`
holds the centrelines and the sized radii. Sewing that isosurface produces a
STEP file with on the order of a million planar faces — minutes to hours of
OCC time, hundreds of megabytes, and a body no CAD system will open — to
approximate cylinders that could have been written down.

So for the strut families this module goes straight from the centrelines to
geometry:

``3MF beam lattice``
    Nodes and beams with per-end radii, in the format the AM tool chain
    already reads (the 3MF Beam Lattice extension, ISO/ASTM 52915 family).
    Exact, a few hundred kilobytes, and effectively instant, because it stores
    the lattice as the lattice rather than as a tessellation of it. This is
    what the industry actually ships lattices in, and it is the recommended
    output.

``STEP``
    Exact analytic B-rep: one cylinder or truncated cone per member, one ball
    per node, each a real quadric surface rather than a facet. Face count is
    proportional to the member count, not to the voxel count.

What this does not do
---------------------
TPMS sheets have no exact B-rep — a gyroid is not a union of quadrics — and
they keep the isosurface route. :func:`lattice_cad_strategy` is the one place
that decides which result gets which treatment.

Nothing here replaces re-analysis. These are the same centrelines the axial
truss sizing used, so the geometry inherits that model's limits: joint bending,
local stress concentration and as-built defects are not in it.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "BeamLattice",
    "beam_lattice_from_member_plan",
    "beam_lattice_from_structure",
    "beam_lattice_solid",
    "beam_lattice_triangles",
    "lattice_cad_strategy",
    "write_beam_lattice_3mf",
]


# 3MF package layout. The beam lattice extension is a child of <mesh>, so the
# object is an ordinary 3MF object whose triangle list happens to be empty.
_CORE_NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
_BEAM_NS = "http://schemas.microsoft.com/3dmanufacturing/beamlattice/2017/02"
_MODEL_PART = "3D/3dmodel.model"

#: Above this many solids the OCC boolean stops being the right tool. The
#: general fuse grows sharply worse than linearly in argument count -- measured
#: here at 3 s for a 229-solid cubic lattice and 105 s for a 564-solid BCC one
#: -- and the BCC case also came back *invalid*, with 81 faces where it should
#: have had hundreds, because eight members meeting at a shallow angle on the
#: body-centre node is exactly the configuration a general fuse handles worst.
#: The union is therefore opt-in, capped, and checked before it is trusted.
DEFAULT_MAXIMUM_FUSED_SOLIDS = 250


@dataclass(frozen=True)
class BeamLattice:
    """A strut lattice as centrelines and radii, in model units.

    ``nodes`` are model-space points, ``beams`` index pairs into them, and
    ``radii`` carries one radius per beam end so a de-homogenized lattice can
    taper a member between two different local densities.
    """

    nodes: np.ndarray
    beams: np.ndarray
    radii: np.ndarray
    unit: str = "millimeter"
    family: str = ""
    source: str = ""

    def __post_init__(self) -> None:
        nodes = np.asarray(self.nodes, dtype=float).reshape((-1, 3))
        beams = np.asarray(self.beams, dtype=np.int64).reshape((-1, 2))
        radii = np.asarray(self.radii, dtype=float)
        if radii.ndim == 1:
            radii = np.repeat(radii[:, None], 2, axis=1)
        radii = radii.reshape((-1, 2))
        if len(beams) != len(radii):
            raise ValueError("A beam lattice needs one radius pair per beam.")
        if len(beams) and (
            beams.min() < 0 or beams.max() >= len(nodes)
        ):
            raise ValueError("Beam lattice member indices are out of range.")
        if not np.all(np.isfinite(nodes)) or not np.all(np.isfinite(radii)):
            raise ValueError("Beam lattice coordinates and radii must be finite.")
        if len(radii) and np.any(radii <= 0.0):
            raise ValueError("Beam lattice radii must be positive.")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "beams", beams)
        object.__setattr__(self, "radii", radii)

    @property
    def lengths(self) -> np.ndarray:
        if not len(self.beams):
            return np.zeros(0, dtype=float)
        delta = self.nodes[self.beams[:, 1]] - self.nodes[self.beams[:, 0]]
        return np.linalg.norm(delta, axis=1)

    @property
    def node_radii(self) -> np.ndarray:
        """Largest radius meeting each node, for sizing the joint ball."""
        radii = np.zeros(len(self.nodes), dtype=float)
        for end in range(2):
            np.maximum.at(radii, self.beams[:, end], self.radii[:, end])
        return radii

    def solid_volume(self) -> float:
        """Volume of the members alone, ignoring the overlap at the joints.

        An upper bound on the built volume, and deliberately labelled as one:
        every joint double-counts the intersection of the members meeting
        there, which is exactly why the built mass has to be measured on the
        rasterized field rather than summed from the centrelines.
        """
        lengths = self.lengths
        r1, r2 = self.radii[:, 0], self.radii[:, 1]
        # Frustum volume; equals the cylinder when the two radii agree.
        return float(
            np.sum(math.pi * lengths * (r1 * r1 + r1 * r2 + r2 * r2) / 3.0)
        )

    def solid_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Bounding box of the members, radii included.

        Not the same as the bounding box of the nodes. The outermost node of a
        clipped network sits at a voxel *centre*, so a member centred there
        reaches half a pitch further out than the grid does, plus its radius.
        The voxel rasterization truncates that overhang at the envelope; an
        analytic member has no such edge to be truncated at, and the two
        representations differ there by up to one radius. It is small and it is
        real, so it gets measured and reported rather than trimmed: an OCC
        boolean against the domain box was tried and deleted three quarters of
        the lattice, which is a far worse answer than a stated overhang.
        """
        used = np.unique(self.beams) if len(self.beams) else np.zeros(0, int)
        if not used.size:
            zero = np.zeros(3, dtype=float)
            return zero, zero
        radii = self.node_radii[used][:, None]
        points = self.nodes[used]
        return (points - radii).min(axis=0), (points + radii).max(axis=0)

    def diagnostics(self) -> dict[str, Any]:
        lengths = self.lengths
        lower, upper = self.solid_bounds()
        return {
            "solid_bounds_min": [float(v) for v in lower],
            "solid_bounds_max": [float(v) for v in upper],
            "representation": "analytic beam centrelines",
            "family": self.family,
            "source": self.source,
            "unit": self.unit,
            "node_count": int(len(self.nodes)),
            "beam_count": int(len(self.beams)),
            "minimum_diameter": float(2.0 * self.radii.min())
            if len(self.radii)
            else 0.0,
            "maximum_diameter": float(2.0 * self.radii.max())
            if len(self.radii)
            else 0.0,
            "minimum_length": float(lengths.min()) if len(lengths) else 0.0,
            "maximum_length": float(lengths.max()) if len(lengths) else 0.0,
            "member_volume_upper_bound": self.solid_volume(),
        }


# ── building a lattice from a solved study ────────────────────────────────────


def _grid_to_model(
    fractional: np.ndarray,
    grid_shape: tuple[int, int, int],
    bounds: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, float]:
    """Map fractional grid coordinates to model space.

    Returns the points and the mean voxel pitch, which is what a radius stated
    in voxels has to be multiplied by. The mapping is the same one surface
    recovery uses — voxel *centres*, not corners — so an analytic body and a
    recovered mesh from the same solve land on each other.
    """
    shape = np.asarray(grid_shape, dtype=float)
    mins = np.asarray(bounds[0], dtype=float)[:3]
    maxs = np.asarray(bounds[1], dtype=float)[:3]
    spacing = (maxs - mins) / np.maximum(shape, 1.0)
    index = np.asarray(fractional, dtype=float) * np.maximum(shape - 1.0, 1.0)
    return mins + (index + 0.5) * spacing, float(np.mean(spacing))


def beam_lattice_from_member_plan(
    plan: Any,
    bounds: tuple[np.ndarray, np.ndarray],
    *,
    unit: str = "millimeter",
) -> BeamLattice:
    """Convert a sized member plan into a model-space beam lattice.

    This is the direct case: the truss sizing stage already produced the
    centrelines and one radius per member against stress and buckling, so the
    CAD body is a change of coordinates away.
    """
    nodes, pitch = _grid_to_model(
        plan.nodes_fractional, tuple(plan.grid_shape), bounds
    )
    radii = np.asarray(plan.radii_voxels, dtype=float) * pitch
    return BeamLattice(
        nodes=nodes,
        beams=np.asarray(plan.members, dtype=np.int64),
        radii=np.maximum(radii, 1e-9),
        unit=unit,
        family=str(plan.mode),
        source="axial truss fully-stressed member sizing",
    )


def beam_lattice_from_structure(
    density: np.ndarray,
    cutoff: float,
    options: Any,
    bounds: tuple[np.ndarray, np.ndarray],
    *,
    passive_solid_mask: np.ndarray | None = None,
    passive_void_mask: np.ndarray | None = None,
    unit: str = "millimeter",
) -> BeamLattice:
    """Build a beam lattice for a strut family that was not member-sized.

    The network comes from the same registry edge list the voxel rasterizer
    uses, clipped to the optimized envelope. Each member's radius is read from
    the de-homogenization map at the optimizer's own density, sampled at that
    member's two ends — so the built member is the one the macro solve was
    driven by, and it grades along its length exactly as the density does.
    """
    from ..manufacturing import lattice_families as families
    from ..manufacturing.member_sizing import (
        _clip_network_to_envelope,
        _connect_clipped_network,
    )
    from ..manufacturing.structures import _strut_radius_field

    family = families.FAMILIES.get(options.mode)
    if family is None or not family.is_strut:
        raise ValueError(
            "An analytic beam lattice needs a strut family; "
            f"{options.display_name} is not one."
        )

    field = np.nan_to_num(np.asarray(density, dtype=float), nan=0.0)
    envelope = field >= float(np.clip(cutoff, 1e-6, 0.999999))
    if passive_solid_mask is not None:
        envelope = envelope | np.asarray(passive_solid_mask, dtype=bool)
    if passive_void_mask is not None:
        envelope = envelope & ~np.asarray(passive_void_mask, dtype=bool)
    if not np.any(envelope):
        raise ValueError("The optimized envelope is empty; there is no lattice.")

    shape = tuple(int(v) for v in field.shape)
    cell_size = max(3.0, float(options.cell_size_voxels))
    cell_counts = tuple(
        int(max(1, math.ceil(extent / cell_size))) for extent in shape
    )
    nodes_fractional, members = families.strut_network(options.mode, cell_counts)
    nodes_fractional, members = _clip_network_to_envelope(
        envelope, nodes_fractional, members
    )
    if len(members) < 1:
        raise ValueError(
            "No lattice member survived clipping to the optimized envelope. "
            "Reduce the cell pitch or refine the analysis grid."
        )
    # Trimming deletes every member that straddles the boundary, which leaves
    # the survivors in islands wherever the envelope is thinner than the cell.
    # Rejoin them inside the envelope so the displayed and exported body is the
    # connected structure the density field describes.
    members, reconnecting_members = _connect_clipped_network(
        envelope, nodes_fractional, members, cell_size
    )
    if reconnecting_members:
        logger.info(
            "Added %d connecting member(s) to rejoin a %s network that "
            "trimming to the optimized envelope had broken into islands.",
            reconnecting_members,
            options.display_name,
        )

    # Radius per node from the local optimized density, then per member end.
    index = np.rint(
        np.clip(nodes_fractional, 0.0, 1.0)
        * (np.asarray(shape, dtype=float) - 1.0)
    ).astype(int)
    local = field[index[:, 0], index[:, 1], index[:, 2]]
    if options.variable_density:
        local = np.clip(
            local,
            options.minimum_relative_density,
            options.maximum_relative_density,
        )
        node_radius_voxels = np.asarray(
            _strut_radius_field(
                options.member_thickness_voxels,
                local,
                mode=options.mode,
                cell_size=cell_size,
                member_scale=options.member_scale,
            ),
            dtype=float,
        )
    else:
        node_radius_voxels = np.full(
            len(nodes_fractional),
            max(0.25, 0.5 * float(options.member_thickness_voxels)),
            dtype=float,
        )

    nodes, pitch = _grid_to_model(nodes_fractional, shape, bounds)
    radii = node_radius_voxels[members] * pitch
    return BeamLattice(
        nodes=nodes,
        beams=members,
        radii=np.maximum(radii, 1e-9),
        unit=unit,
        family=options.mode,
        source=(
            "de-homogenized from the optimized density field"
            if options.variable_density
            else "uniform member thickness"
        ),
    )


def lattice_cad_strategy(options: Any) -> str:
    """Return ``"beam"``, ``"isosurface"`` or ``"solid"`` for a structure mode.

    One predicate, so the export path, the report and the UI cannot disagree
    about whether a given study has an exact analytic body available.
    """
    from ..manufacturing.lattice_families import FAMILIES

    mode = getattr(options, "mode", "") if options is not None else ""
    if not mode or mode == "solid":
        return "solid"
    family = FAMILIES.get(mode)
    if family is not None and family.is_strut:
        return "beam"
    return "isosurface"


# ── 3MF beam lattice ──────────────────────────────────────────────────────────


_CONTENT_TYPES = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-'
    'package.relationships+xml"/>'
    '<Default Extension="model" ContentType="application/vnd.ms-package.'
    '3dmanufacturing-3dmodel+xml"/>'
    "</Types>"
)

_RELATIONSHIPS = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/'
    'relationships">'
    '<Relationship Id="rel0" Target="/3D/3dmodel.model" '
    'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>'
    "</Relationships>"
)


def _model_xml(lattice: BeamLattice) -> Iterable[str]:
    """Yield the 3MF model part in chunks, so a large lattice is not joined.

    A hundred thousand beams is an ordinary size here and a single joined
    string of it is a needless hundreds-of-megabytes allocation.
    """
    radii = lattice.radii
    default_radius = float(np.median(radii)) if len(radii) else 1.0
    lengths = lattice.lengths
    # `minlength` is the shortest member actually present, which is what the
    # specification asks for and what a consumer uses to decide its vertex
    # welding tolerance. Nudged below the measured value so a beam written at
    # `%.6g` cannot round to just under its own declared minimum.
    minimum_length = float(lengths.min()) if len(lengths) else 1.0
    minimum_length = max(minimum_length * (1.0 - 1e-6), 1e-9)

    yield (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<model unit="{lattice.unit}" xml:lang="en-US" '
        f'xmlns="{_CORE_NS}" xmlns:b="{_BEAM_NS}" requiredextensions="b">'
        "<resources>"
        '<object id="1" type="model">'
        "<mesh>"
        "<vertices>"
    )
    for point in lattice.nodes:
        yield '<vertex x="{:.6g}" y="{:.6g}" z="{:.6g}"/>'.format(*point)
    # The core specification requires the element; a beam lattice object
    # legitimately has no surface triangles of its own.
    yield "</vertices><triangles/>"
    yield (
        f'<b:beamlattice radius="{default_radius:.6g}" '
        f'minlength="{minimum_length:.6g}" cap="sphere">'
        "<b:beams>"
    )
    for (start, end), (r1, r2) in zip(lattice.beams, radii):
        yield (
            f'<b:beam v1="{int(start)}" v2="{int(end)}" '
            f'r1="{float(r1):.6g}" r2="{float(r2):.6g}"/>'
        )
    yield (
        "</b:beams></b:beamlattice>"
        "</mesh></object></resources>"
        '<build><item objectid="1"/></build>'
        "</model>"
    )


def write_beam_lattice_3mf(path: str | Path, lattice: BeamLattice) -> Path:
    """Write a lattice as a 3MF beam lattice package.

    The exact geometry, in the format the AM tool chain reads natively, at a
    size proportional to the member count. No isosurface, no mesh repair and no
    B-rep sewing is involved, because none of them is needed to say where a
    cylinder is.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as package:
        package.writestr("[Content_Types].xml", _CONTENT_TYPES)
        package.writestr("_rels/.rels", _RELATIONSHIPS)
        with package.open(_MODEL_PART, "w") as stream:
            for chunk in _model_xml(lattice):
                stream.write(chunk.encode("utf-8"))
    logger.info(
        "Wrote %d beams and %d nodes to %s (%.1f kB).",
        len(lattice.beams),
        len(lattice.nodes),
        target,
        target.stat().st_size / 1024.0,
    )
    return target


# ── analytic B-rep ────────────────────────────────────────────────────────────


def _member_solid(
    start: np.ndarray,
    end: np.ndarray,
    radius_start: float,
    radius_end: float,
) -> Any:
    """One member as an exact quadric solid: a cylinder, or a cone if tapered."""
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCone, BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

    direction = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    length = float(np.linalg.norm(direction))
    if length <= 1e-12:
        raise ValueError("A lattice member has zero length.")
    axis = gp_Ax2(
        gp_Pnt(*(float(v) for v in start)),
        gp_Dir(*(float(v) / length for v in direction)),
    )
    if abs(float(radius_end) - float(radius_start)) <= 1e-9 * length:
        return BRepPrimAPI_MakeCylinder(
            axis, float(radius_start), length
        ).Shape()
    return BRepPrimAPI_MakeCone(
        axis, float(radius_start), float(radius_end), length
    ).Shape()


def beam_lattice_solid(
    lattice: BeamLattice,
    *,
    node_balls: bool = True,
    fuse: bool = False,
    maximum_fused_solids: int = DEFAULT_MAXIMUM_FUSED_SOLIDS,
    fuzzy_value: float = 0.0,
    return_report: bool = False,
) -> Any:
    """Build an exact analytic B-rep body from the centrelines.

    One cylinder or truncated cone per member and one ball per joint, so every
    face is a real quadric and the face count tracks the member count rather
    than the voxel count. Building the solids is the fast part and is always
    exact.

    ``fuse`` additionally unions them into a single solid. It is off by default
    and deliberately so: on measured lattices the union costs far more than
    everything else in the study put together and, worse, can come back invalid
    without saying so. When it is requested the result is checked and discarded
    if it does not survive; the caller is told which of the two it got, because
    "one solid" and "a set of solids overlapping at the joints" are not the
    same thing to whatever consumes the file next. Both are valid STEP and both
    are exact.
    """
    import cadquery as cq
    from OCP.BRep import BRep_Builder
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeSphere
    from OCP.gp import gp_Pnt
    from OCP.TopoDS import TopoDS_Compound

    if not len(lattice.beams):
        raise ValueError("A beam lattice with no members has no CAD body.")

    solids: list[Any] = []
    skipped = 0
    lengths = lattice.lengths
    for index, (start_id, end_id) in enumerate(lattice.beams):
        if lengths[index] <= 1e-9:
            skipped += 1
            continue
        try:
            solids.append(
                _member_solid(
                    lattice.nodes[int(start_id)],
                    lattice.nodes[int(end_id)],
                    float(lattice.radii[index, 0]),
                    float(lattice.radii[index, 1]),
                )
            )
        except Exception:
            skipped += 1

    ball_count = 0
    if node_balls:
        # A ball at each joint fills the wedge the cylinders leave between them
        # and rounds the junction, which is both the printable geometry and the
        # one place a bare cylinder union has a stress-raising sharp edge.
        radii = lattice.node_radii
        used = np.unique(lattice.beams)
        for node_id in used:
            radius = float(radii[int(node_id)])
            if radius <= 1e-9:
                continue
            point = lattice.nodes[int(node_id)]
            solids.append(
                BRepPrimAPI_MakeSphere(
                    gp_Pnt(*(float(v) for v in point)), radius
                ).Shape()
            )
            ball_count += 1

    if not solids:
        raise ValueError("No lattice member produced a valid solid.")

    report: dict[str, Any] = {
        "method": "Analytic beam lattice",
        "representation": (
            "one cylinder or cone per member, one ball per joint"
        ),
        "editable": True,
        "smooth": True,
        "member_count": int(len(lattice.beams) - skipped),
        "joint_ball_count": int(ball_count),
        "skipped_member_count": int(skipped),
        "solid_count": int(len(solids)),
    }

    fused = None
    if fuse and len(solids) <= int(maximum_fused_solids):
        try:
            candidate = _fuse_solids(solids, fuzzy_value=fuzzy_value)
            if _is_valid_shape(candidate):
                fused = candidate
            else:
                report["fuse_error"] = (
                    "the union produced an invalid B-rep and was discarded"
                )
        except Exception as exc:
            logger.info(
                "Beam lattice fuse failed (%s); returning a compound instead.",
                exc,
            )
            report["fuse_error"] = str(exc)
    elif fuse:
        report["fuse_error"] = (
            f"{len(solids)} solids exceeds the {int(maximum_fused_solids)} the "
            "union is allowed to attempt"
        )

    if fused is not None:
        shape = cq.Shape.cast(fused)
        report["fused"] = True
        report["cad_face_count"] = int(len(shape.Faces()))
    else:
        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        for solid in solids:
            builder.Add(compound, solid)
        shape = cq.Shape.cast(compound)
        report["fused"] = False
        report["cad_face_count"] = int(len(shape.Faces()))
        report["guidance"] = (
            f"{len(solids)} exact solids were written without a boolean union. "
            "They overlap at the joints, which slicers and meshers accept and "
            "which every CAD system can union on import if a single closed "
            "solid is wanted. The 3MF beam lattice export needs no union at "
            "all and is smaller again."
        )

    if return_report:
        return cq.Workplane(obj=shape), report
    return cq.Workplane(obj=shape)


def _is_valid_shape(shape: Any) -> bool:
    """Whether OpenCASCADE considers a shape a valid B-rep."""
    try:
        from OCP.BRepCheck import BRepCheck_Analyzer

        return bool(BRepCheck_Analyzer(shape).IsValid())
    except Exception:
        logger.debug("BRepCheck_Analyzer unavailable; skipping the fuse check")
        return True


def _fuse_solids(solids: list[Any], *, fuzzy_value: float = 0.0) -> Any:
    """Union a list of solids in one batch general-fuse call."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCP.TopTools import TopTools_ListOfShape

    arguments = TopTools_ListOfShape()
    arguments.Append(solids[0])
    tools = TopTools_ListOfShape()
    for solid in solids[1:]:
        tools.Append(solid)

    operation = BRepAlgoAPI_Fuse()
    operation.SetArguments(arguments)
    operation.SetTools(tools)
    operation.SetRunParallel(True)
    if fuzzy_value > 0.0:
        operation.SetFuzzyValue(float(fuzzy_value))
    operation.Build()
    if hasattr(operation, "HasErrors") and operation.HasErrors():
        raise RuntimeError("OCC reported errors while fusing the lattice.")
    if not operation.IsDone():
        raise RuntimeError("OCC could not fuse the lattice members.")
    return operation.Shape()


# ── analytic tessellation ─────────────────────────────────────────────────────


def _cylinder_template(segments: int) -> tuple[np.ndarray, np.ndarray]:
    """Unit capped cylinder along +Z, radius 1, height 1, as (vertices, faces)."""
    count = max(3, int(segments))
    angle = 2.0 * np.pi * np.arange(count, dtype=float) / count
    ring = np.stack([np.cos(angle), np.sin(angle), np.zeros(count)], axis=1)
    bottom = ring.copy()
    top = ring + np.asarray([0.0, 0.0, 1.0])
    vertices = np.concatenate(
        [bottom, top, [[0.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]]], axis=0
    )
    faces: list[tuple[int, int, int]] = []
    for i in range(count):
        j = (i + 1) % count
        faces.append((i, j, count + j))
        faces.append((i, count + j, count + i))
        faces.append((2 * count, j, i))
        faces.append((2 * count + 1, count + i, count + j))
    return vertices, np.asarray(faces, dtype=np.int64)


def beam_lattice_triangles(
    lattice: BeamLattice,
    *,
    segments: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Tessellate the lattice directly, without an isosurface.

    Every member becomes a closed capped prism about its own axis, so the
    struts come out round at any size instead of carrying the recovery grid's
    terracing, and the cost is proportional to the member count. The result is
    a union of overlapping closed shells rather than one manifold surface,
    which is what a slicer wants and is not what a watertight-mesh check wants;
    this feeds preview and STL export, not the validated recovered shape.
    """
    template_vertices, template_faces = _cylinder_template(segments)
    beams = lattice.beams
    if not len(beams):
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=np.int64)

    start = lattice.nodes[beams[:, 0]]
    end = lattice.nodes[beams[:, 1]]
    axis = end - start
    length = np.linalg.norm(axis, axis=1)
    keep = length > 1e-12
    start, end, axis, length = start[keep], end[keep], axis[keep], length[keep]
    radii = lattice.radii[keep]
    direction = axis / length[:, None]

    # An orthonormal frame per member. Choosing the seed axis as the one the
    # member points along least keeps the cross product well conditioned.
    seed = np.zeros_like(direction)
    seed[np.arange(len(direction)), np.argmin(np.abs(direction), axis=1)] = 1.0
    u = np.cross(direction, seed)
    u /= np.linalg.norm(u, axis=1)[:, None]
    v = np.cross(direction, u)

    # Scale the template's unit ring by the radius at each end before rotating.
    local = np.broadcast_to(
        template_vertices, (len(direction),) + template_vertices.shape
    ).copy()
    ring = (len(template_vertices) - 2) // 2
    local[:, :ring, :2] *= radii[:, 0][:, None, None]
    local[:, ring : 2 * ring, :2] *= radii[:, 1][:, None, None]
    local[..., 2] *= length[:, None]

    vertices = (
        start[:, None, :]
        + local[..., 0][..., None] * u[:, None, :]
        + local[..., 1][..., None] * v[:, None, :]
        + local[..., 2][..., None] * direction[:, None, :]
    )
    offsets = (
        np.arange(len(direction), dtype=np.int64)[:, None, None]
        * len(template_vertices)
    )
    faces = template_faces[None, :, :] + offsets
    return (
        vertices.reshape((-1, 3)),
        faces.reshape((-1, 3)),
    )
