# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""One registry for every periodic cell family PyLCSS can build.

Before this module the cell geometry was written out three times — once to
rasterize the part (:mod:`.structures`), once to homogenize a unit cell
(:mod:`.cell_material`), and once to build a sizeable truss
(:mod:`.member_sizing`). Three copies of a unit cell is three chances for the
manufactured geometry to stop being the geometry the optimizer was driven by,
which is the one error in a two-scale workflow that no downstream check can
catch. Everything now reads the definitions here.

Surface families
----------------
The TPMS surfaces are the standard nodal (short Fourier) approximations of the
minimal surfaces. They are approximations of the exact surfaces, accurate to a
few percent in position at the densities a lattice actually operates at, and
they are what every implicit modelling tool in this space evaluates.

Each surface supports two wall interpretations, and they are different
materials, not a rendering choice:

``sheet``
    The solid is the shell around the surface, ``|f| <= t``. Two disjoint
    open channels, no closed cells to trap powder, and the well-known
    near-Hashin-Shtrikman stiffness at low density.
``skeletal``
    The solid is one side of the surface, ``f <= t`` — the "solid network" or
    "ligament" form. One solid phase and one open channel. At equal relative
    density it is markedly more compliant and collapses in a bending-dominated
    mode rather than the sheet's stretch-dominated one, so it is chosen for
    energy absorption and for flow/scaffold work rather than for stiffness.

Strut families
--------------
Given as an edge list over the unit cell ``[0, 1]^3``. ``maxwell`` records
whether the untrimmed periodic network is stretch- or bending-dominated, which
is the property that decides whether the part carries load through axial
member force or through joint bending, and therefore whether the axial truss
sizing in :mod:`.member_sizing` is a valid surrogate at all.

Homogenization
--------------
``homogenized`` marks the families whose periodic cell is a cube *and* whose
effective tensor has cubic symmetry, so three constants describe it exactly and
the part-scale operator stays a three-term sum. Those families drive the
two-scale solve from their measured tensor. The rest keep the isotropic
continuum surrogate and must say so in their report — the same position
honeycomb has always had here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

__all__ = [
    "LatticeFamily",
    "FAMILIES",
    "PUBLIC_FAMILY_KEYS",
    "PUBLIC_FAMILIES",
    "PUBLIC_LATTICE_FAMILY_NAMES",
    "STRUT_FAMILIES",
    "TPMS_FAMILIES",
    "HOMOGENIZED_FAMILIES",
    "MAXIMUM_MEMBER_RADIUS",
    "RASTER_ALLOWANCE_AXIAL",
    "RASTER_ALLOWANCE_DIAGONAL",
    "family_for",
    "normalize_family_key",
    "strut_distance_field",
    "strut_edges",
    "strut_network",
    "strut_occupancy",
    "strut_raster_margin",
    "strut_unit_cell_distance",
    "tpms_implicit_field",
    "tpms_specific_surface_area",
    "tpms_unit_cell_field",
]


# ── surface definitions ───────────────────────────────────────────────────────
#
# Every function takes the three phase arrays (2*pi*position/pitch) and returns
# the nodal approximation of the level-set function. Written to broadcast, so
# the caller may pass 1-D axes shaped (n,1,1), (1,n,1), (1,1,n) and get the
# full field without ever materializing three 3-D coordinate arrays. That is
# not a micro-optimization: on a 400^3 recovery grid the coordinate arrays
# alone are 1.5 GB, and the separable form removes them.


def _gyroid(x, y, z):
    return (
        np.sin(x) * np.cos(y)
        + np.sin(y) * np.cos(z)
        + np.sin(z) * np.cos(x)
    )


def _diamond(x, y, z):
    return (
        np.sin(x) * np.sin(y) * np.sin(z)
        + np.sin(x) * np.cos(y) * np.cos(z)
        + np.cos(x) * np.sin(y) * np.cos(z)
        + np.cos(x) * np.cos(y) * np.sin(z)
    )


def _primitive(x, y, z):
    return np.cos(x) + np.cos(y) + np.cos(z)


def _iwp(x, y, z):
    return 2.0 * (
        np.cos(x) * np.cos(y)
        + np.cos(y) * np.cos(z)
        + np.cos(z) * np.cos(x)
    ) - (np.cos(2.0 * x) + np.cos(2.0 * y) + np.cos(2.0 * z))


def _neovius(x, y, z):
    return 3.0 * (np.cos(x) + np.cos(y) + np.cos(z)) + 4.0 * (
        np.cos(x) * np.cos(y) * np.cos(z)
    )


def _fischer_koch_s(x, y, z):
    return (
        np.cos(2.0 * x) * np.sin(y) * np.cos(z)
        + np.cos(x) * np.cos(2.0 * y) * np.sin(z)
        + np.sin(x) * np.cos(y) * np.cos(2.0 * z)
    )


def _lidinoid(x, y, z):
    return (
        0.5
        * (
            np.sin(2.0 * x) * np.cos(y) * np.sin(z)
            + np.sin(2.0 * y) * np.cos(z) * np.sin(x)
            + np.sin(2.0 * z) * np.cos(x) * np.sin(y)
        )
        - 0.5
        * (
            np.cos(2.0 * x) * np.cos(2.0 * y)
            + np.cos(2.0 * y) * np.cos(2.0 * z)
            + np.cos(2.0 * z) * np.cos(2.0 * x)
        )
        + 0.15
    )


def _split_p(x, y, z):
    return (
        1.1
        * (
            np.sin(2.0 * x) * np.sin(z) * np.cos(y)
            + np.sin(2.0 * y) * np.sin(x) * np.cos(z)
            + np.sin(2.0 * z) * np.sin(y) * np.cos(x)
        )
        - 0.2
        * (
            np.cos(2.0 * x) * np.cos(2.0 * y)
            + np.cos(2.0 * y) * np.cos(2.0 * z)
            + np.cos(2.0 * z) * np.cos(2.0 * x)
        )
        - 0.4 * (np.cos(2.0 * x) + np.cos(2.0 * y) + np.cos(2.0 * z))
    )


_SURFACES: dict[str, Callable[..., np.ndarray]] = {
    "gyroid": _gyroid,
    "diamond": _diamond,
    "primitive": _primitive,
    "iwp": _iwp,
    "neovius": _neovius,
    "fischer_koch": _fischer_koch_s,
    "lidinoid": _lidinoid,
    "split_p": _split_p,
}


# ── strut definitions ─────────────────────────────────────────────────────────


def _cube_corners() -> list[tuple[float, float, float]]:
    return [
        (x, y, z)
        for x in (0.0, 1.0)
        for y in (0.0, 1.0)
        for z in (0.0, 1.0)
    ]


def _cube_edges() -> list[tuple[tuple, tuple]]:
    edges: list[tuple[tuple, tuple]] = []
    for corner in _cube_corners():
        for axis in range(3):
            if corner[axis] != 0.0:
                continue
            other = list(corner)
            other[axis] = 1.0
            edges.append((corner, tuple(other)))
    return edges


def _z_pillars() -> list[tuple[tuple, tuple]]:
    return [
        ((x, y, 0.0), (x, y, 1.0))
        for x in (0.0, 1.0)
        for y in (0.0, 1.0)
    ]


def _bcc_edges() -> list[tuple[tuple, tuple]]:
    center = (0.5, 0.5, 0.5)
    return [(center, corner) for corner in _cube_corners()]


def _octet_edges() -> list[tuple[tuple, tuple]]:
    """The FCC bond graph: every site to all twelve nearest neighbours.

    Tiled, this is the octet truss — an octahedron of face-centre-to-face-centre
    struts interpenetrating tetrahedra of corner-to-face-centre struts. Its
    defining property is that every node is *similarly situated* with nodal
    connectivity 12, which is the lower limit for a stretch-dominated 3-D
    lattice (Deshpande, Fleck & Ashby 2001) and the reason this family carries
    the axial-truss member sizing at all.

    Emitting only the corner-to-face-centre bonds is the failure worth naming,
    because it looks right: it draws the tetrahedra, tiles periodically, and
    rasterizes into something that photographs like an octet truss. But it
    leaves every face centre with connectivity 4 instead of 12 — a mechanism by
    Maxwell's count — so the cell is bending dominated, and the homogenized law
    measured from it came out at ``E/E0 ~ rho^1.47`` where the octet truss is
    linear in density (``E/Es = rho/9``). A macro solve driven by that tensor is
    solving a different material than the one the part is built from.

    Written over the four-site FCC basis rather than over all fourteen sites of
    the cube: each basis site contributes its full twelve bonds, so every bond
    in the periodic network appears at least once, and the tiling in
    :func:`strut_network` and :func:`_tiled_edges` deduplicates the rest.
    """
    basis = np.asarray(
        [(0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)],
        dtype=float,
    )
    # Neighbour candidates: the same basis replicated over the surrounding
    # cells, which is where a nearest neighbour of a cell-boundary site lives.
    offsets = np.asarray(
        [
            (i, j, k)
            for i in (-1.0, 0.0, 1.0)
            for j in (-1.0, 0.0, 1.0)
            for k in (-1.0, 0.0, 1.0)
        ],
        dtype=float,
    )
    candidates = (basis[None, :, :] + offsets[:, None, :]).reshape(-1, 3)
    spacing = 1.0 / np.sqrt(2.0)

    edges: list[tuple[tuple, tuple]] = []
    seen: set[tuple[int, ...]] = set()
    for site in basis:
        distance = np.linalg.norm(candidates - site, axis=1)
        for index in np.nonzero(np.abs(distance - spacing) < 1e-9)[0]:
            neighbour = candidates[index]
            ends = sorted(
                tuple(int(round(float(v) * 1_000_000)) for v in point)
                for point in (site, neighbour)
            )
            key = tuple(ends[0]) + tuple(ends[1])
            if key in seen:
                continue
            seen.add(key)
            edges.append(
                (
                    tuple(float(v) for v in site),
                    tuple(float(v) for v in neighbour),
                )
            )
    return edges


def _diamond_truss_edges() -> list[tuple[tuple, tuple]]:
    """The diamond-cubic bond graph: tetrahedral, connectivity 4.

    Four ``B`` sites at the FCC positions offset by (1/4, 1/4, 1/4), each bonded
    to its four nearest ``A`` sites. Bending dominated by Maxwell's count, and
    the open, self-supporting strut analogue of the Schwarz D sheet.
    """
    a_sites = np.asarray(
        [(0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)],
        dtype=float,
    )
    # Replicate the A sublattice over the neighbouring cells so a B site near a
    # cell face still finds all four of its bonds.
    offsets = np.asarray(
        [(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=float
    )
    neighbours = (a_sites[None, :, :] + offsets[:, None, :]).reshape(-1, 3)
    edges: list[tuple[tuple, tuple]] = []
    for site in a_sites:
        b = site + 0.25
        distance = np.linalg.norm(neighbours - b, axis=1)
        for index in np.argsort(distance)[:4]:
            edges.append(
                (
                    tuple(float(v) for v in b),
                    tuple(float(v) for v in neighbours[index]),
                )
            )
    return edges


def _strut_edge_table() -> dict[str, np.ndarray]:
    table = {
        "cubic": _cube_edges(),
        "bcc": _bcc_edges(),
        "bccz": _bcc_edges() + _z_pillars(),
        "octet": _octet_edges(),
        "fccz": _octet_edges() + _z_pillars(),
        "diamond_truss": _diamond_truss_edges(),
    }
    return {
        key: np.asarray(edges, dtype=float).reshape((-1, 2, 3))
        for key, edges in table.items()
    }


_STRUT_EDGES = _strut_edge_table()


# ── the registry ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LatticeFamily:
    """Everything the rest of PyLCSS needs to know about one cell family."""

    key: str
    display_name: str
    #: ``tpms``, ``strut`` or ``prismatic``.
    kind: str
    #: Underlying surface or strut-graph key; several families share one.
    base: str
    #: ``sheet`` or ``skeletal`` for TPMS, otherwise empty.
    wall_mode: str = ""
    #: Cubic effective tensor, so a measured three-constant law is exact.
    homogenized: bool = False
    #: Untrimmed periodic deformation mode.
    maxwell: str = ""
    #: Nodal connectivity of a strut family; 0 where it does not apply.
    connectivity: int = 0
    #: Voxels per cell the *resolved* build grid needs before this family's
    #: geometry stays connected at a working 20% relative density. Measured by
    #: rasterizing one family at a sweep of resolutions and taking the largest
    #: face-connected component as a fraction of the solid: below the value
    #: here the surface pinches off at its own necks and the connectivity cull
    #: legitimately deletes most of it. The recovery grid is supersampled, so a
    #: cell of 8 analysis voxels is usually 24 to 80 resolved ones — this is
    #: compared against the resolved number, never the requested one.
    minimum_cell_voxels: float = 6.0
    notes: str = ""

    @property
    def is_tpms(self) -> bool:
        return self.kind == "tpms"

    @property
    def is_strut(self) -> bool:
        return self.kind == "strut"

    @property
    def implicit(self) -> Callable[..., np.ndarray] | None:
        return _SURFACES.get(self.base)


def _tpms(
    key: str,
    display: str,
    base: str,
    wall_mode: str,
    *,
    homogenized: bool,
    maxwell: str,
    minimum_cell_voxels: float,
    notes: str,
) -> LatticeFamily:
    return LatticeFamily(
        key=key,
        display_name=display,
        kind="tpms",
        base=base,
        wall_mode=wall_mode,
        homogenized=homogenized,
        maxwell=maxwell,
        minimum_cell_voxels=minimum_cell_voxels,
        notes=notes,
    )


def _strut(
    key: str,
    display: str,
    *,
    maxwell: str,
    connectivity: int,
    minimum_cell_voxels: float,
    homogenized: bool = True,
    notes: str = "",
) -> LatticeFamily:
    return LatticeFamily(
        key=key,
        display_name=display,
        kind="strut",
        base=key,
        homogenized=homogenized,
        maxwell=maxwell,
        connectivity=connectivity,
        minimum_cell_voxels=minimum_cell_voxels,
        notes=notes,
    )


_FAMILY_LIST: tuple[LatticeFamily, ...] = (
    # ── sheet TPMS ────────────────────────────────────────────────────────────
    _tpms(
        "gyroid", "Gyroid Lattice", "gyroid", "sheet",
        homogenized=True,
        maxwell="stretch",
        minimum_cell_voxels=8.0,
        notes=(
            "Ia-3d minimal surface. No straight channel through the cell in any "
            "direction and its continuously curved walls are generally favorable "
            "for additive manufacture. The process, build direction, wall "
            "thickness and trimmed part boundary still require printability and "
            "powder-evacuation checks. The default printed infill."
        ),
    ),
    _tpms(
        "diamond", "Diamond Lattice", "diamond", "sheet",
        homogenized=True,
        maxwell="stretch",
        minimum_cell_voxels=20.0,
        notes=(
            "Schwarz D. Stiffer than the gyroid at equal density and with a "
            "higher shear modulus, at the cost of steeper overhangs."
        ),
    ),
    _tpms(
        "primitive", "Schwarz Primitive Lattice", "primitive", "sheet",
        homogenized=True,
        maxwell="bending",
        minimum_cell_voxels=8.0,
        notes=(
            "Im-3m. Straight open channels along all three axes, which is why "
            "it dominates heat-exchanger and flow work. Structurally the most "
            "compliant of the common surfaces and it needs support on its "
            "near-horizontal saddles."
        ),
    ),
    _tpms(
        "iwp", "Schoen I-WP Lattice", "iwp", "sheet",
        homogenized=True,
        maxwell="stretch",
        minimum_cell_voxels=20.0,
        notes=(
            "Im-3m. The stiffest of the common sheet surfaces in bulk modulus, "
            "with pronounced cubic anisotropy — the measured tensor matters "
            "more here than for any other family."
        ),
    ),
    _tpms(
        "neovius", "Neovius Lattice", "neovius", "sheet",
        homogenized=True,
        maxwell="stretch",
        minimum_cell_voxels=16.0,
        notes=(
            "Im-3m. High stiffness and high surface area per volume; the "
            "narrow necks make it the hardest of these to print cleanly at low "
            "relative density."
        ),
    ),
    _tpms(
        "fischer_koch", "Fischer-Koch S Lattice", "fischer_koch", "sheet",
        homogenized=False,
        maxwell="stretch",
        minimum_cell_voxels=24.0,
        notes=(
            "Tetragonal, not cubic, so three constants cannot describe it. It "
            "builds and prints, but the part-scale solve falls back to the "
            "isotropic continuum surrogate and the report says so."
        ),
    ),
    _tpms(
        "lidinoid", "Lidinoid Lattice", "lidinoid", "sheet",
        homogenized=True,
        maxwell="stretch",
        minimum_cell_voxels=24.0,
        notes=(
            "I4(1)32, a gyroid relative. Its nodal approximation has the "
            "narrowest necks of the set, so it wants a coarse cell pitch on a "
            "fine grid."
        ),
    ),
    _tpms(
        "split_p", "Split-P Lattice", "split_p", "sheet",
        homogenized=False,
        maxwell="stretch",
        minimum_cell_voxels=20.0,
        notes=(
            "A split gyroid variant. The nodal approximation is poor enough far "
            "from mid density that it keeps the continuum surrogate."
        ),
    ),
    # ── skeletal (solid-network) TPMS ─────────────────────────────────────────
    #
    # Only the surfaces whose two labyrinths are congruent are offered here.
    # For those, either side of f = 0 is the same solid and a requested density
    # gives one connected network. It is not a general property: measured at
    # 20% relative density and 12 to 28 voxels per cell, the Schwarz P and
    # Neovius one-sided solids break into isolated blobs on *both* sides — the
    # P lower set is a single ball at the cell centre and its upper set is
    # eight corner blobs — and the Lidinoid gives two interpenetrating halves
    # rather than one network. Shipping those as "skeletal" would hand the user
    # a pile of disconnected balls with a plausible relative density on it.
    _tpms(
        "gyroid_skeletal", "Gyroid Network (Skeletal)", "gyroid", "skeletal",
        homogenized=True,
        maxwell="bending",
        minimum_cell_voxels=8.0,
        notes=(
            "One solid side of the gyroid instead of a shell around it. Around "
            "half the stiffness of the sheet form at equal density, with a long "
            "flat crush plateau — the reason it is chosen for energy absorption "
            "and for bone-scaffold work rather than for stiffness."
        ),
    ),
    _tpms(
        "diamond_skeletal", "Diamond Network (Skeletal)", "diamond", "skeletal",
        homogenized=True,
        maxwell="bending",
        minimum_cell_voxels=8.0,
        notes=(
            "Solid-network Schwarz D: four ligaments meeting at a tetrahedral "
            "node, the smooth-filleted relative of the diamond truss."
        ),
    ),
    _tpms(
        "iwp_skeletal", "I-WP Network (Skeletal)", "iwp", "skeletal",
        homogenized=True,
        maxwell="bending",
        minimum_cell_voxels=8.0,
        notes=(
            "Solid-network Schoen I-WP: the stiffest of the three skeletal "
            "networks, and strongly cubic-anisotropic."
        ),
    ),
    # ── strut lattices ────────────────────────────────────────────────────────
    _strut(
        "cubic", "Cubic Lattice",
        maxwell="bending", connectivity=6,
        minimum_cell_voxels=6,
        notes=(
            "Simple cubic. Stiff along the three axes and close to a mechanism "
            "in shear, so it is only defensible where the load is axis-aligned."
        ),
    ),
    _strut(
        "bcc", "BCC Lattice",
        maxwell="bending", connectivity=8,
        minimum_cell_voxels=10,
        notes=(
            "Body-centred cubic: the most-printed metal AM strut cell. No "
            "horizontal members in the untrimmed cell, which is favorable for "
            "support reduction. Boundary trimming and the selected build "
            "direction still require an overhang check. It is bending dominated "
            "— compliant, but forgiving and predictable in crush."
        ),
    ),
    _strut(
        "bccz", "BCC-Z Lattice",
        maxwell="mixed", connectivity=10,
        minimum_cell_voxels=12,
        notes=(
            "BCC with vertical pillars added. The pillars carry the axial load "
            "and roughly double the Z stiffness over BCC, at the price of "
            "strong anisotropy about that axis."
        ),
    ),
    _strut(
        "octet", "Octet Truss Lattice",
        maxwell="stretch", connectivity=12,
        minimum_cell_voxels=16,
        notes=(
            "An octahedron of face-centre struts interpenetrating corner "
            "tetrahedra, on the FCC bond graph. Stretch dominated at "
            "connectivity 12 and the stiffest strut cell per unit mass — the "
            "reference for structural strut lattices. Its measured tensor is "
            "cubic but not isotropic at the resolutions homogenized here; the "
            "Zener ratio runs about 1.5 at 30% relative density, so the "
            "measured law matters rather than an isotropic surrogate."
        ),
    ),
    _strut(
        "fccz", "FCC-Z Lattice",
        maxwell="stretch", connectivity=14,
        minimum_cell_voxels=20,
        notes=(
            "Octet with vertical pillars. Higher axial stiffness than octet "
            "along Z; the horizontal-free strut set keeps it printable."
        ),
    ),
    _strut(
        "diamond_truss", "Diamond Truss Lattice",
        maxwell="bending", connectivity=4,
        minimum_cell_voxels=10,
        notes=(
            "The diamond-cubic bond graph. Only four struts per node, so it is "
            "the most open and the most compliant of the strut cells; used "
            "where permeability matters more than stiffness."
        ),
    ),
    # ── prismatic ─────────────────────────────────────────────────────────────
    LatticeFamily(
        key="honeycomb",
        display_name="Honeycomb Lattice",
        kind="prismatic",
        base="honeycomb",
        homogenized=False,
        maxwell="stretch",
        notes=(
            "Extruded hexagonal prisms along Z. Its periodic cell is not a cube "
            "and its tensor is strongly orthotropic, so three constants cannot "
            "describe it and it keeps the continuum surrogate."
        ),
    ),
)


FAMILIES: dict[str, LatticeFamily] = {family.key: family for family in _FAMILY_LIST}

# The complete registry remains available so projects authored with earlier
# PyLCSS releases still deserialize and rebuild exactly as saved. New studies
# deliberately offer a much smaller catalogue: these five cells cover the
# common structural, flow/thermal and prismatic use cases, have comparatively
# mature geometry paths here, and avoid presenting experimental variants as
# equivalent production choices.
PUBLIC_FAMILY_KEYS: tuple[str, ...] = (
    "gyroid",
    "primitive",
    "bcc",
    "octet",
    "honeycomb",
)
PUBLIC_FAMILIES: dict[str, LatticeFamily] = {
    key: FAMILIES[key] for key in PUBLIC_FAMILY_KEYS
}
PUBLIC_LATTICE_FAMILY_NAMES: tuple[str, ...] = tuple(
    PUBLIC_FAMILIES[key].display_name for key in PUBLIC_FAMILY_KEYS
)

TPMS_FAMILIES: tuple[str, ...] = tuple(
    key for key, family in FAMILIES.items() if family.is_tpms
)
STRUT_FAMILIES: tuple[str, ...] = tuple(
    key for key, family in FAMILIES.items() if family.is_strut
)
HOMOGENIZED_FAMILIES: tuple[str, ...] = tuple(
    key for key, family in FAMILIES.items() if family.homogenized
)


_ALIASES: dict[str, str] = {
    "": "solid",
    "none": "solid",
    "solid": "solid",
    "solid envelope": "solid",
    "schwarz p": "primitive",
    "schwarz primitive": "primitive",
    "primitive": "primitive",
    "p": "primitive",
    "schwarz d": "diamond",
    "schoen iwp": "iwp",
    "schoen i wp": "iwp",
    "i wp": "iwp",
    "iwp": "iwp",
    "fischer koch": "fischer_koch",
    "fischer koch s": "fischer_koch",
    "split p": "split_p",
    "octet truss": "octet",
    "bcc z": "bccz",
    "fcc z": "fccz",
    "fcc": "octet",
    "diamond truss": "diamond_truss",
    "gyroid network": "gyroid_skeletal",
    "diamond network": "diamond_skeletal",
    "i wp network": "iwp_skeletal",
    "iwp network": "iwp_skeletal",
}


def _flatten(value: str) -> str:
    return str(value or "").strip().lower().replace("-", " ").replace("_", " ")


#: Display strings are what the inspector stores in the graph property, so they
#: are the primary key a saved project is reopened with. Matching them exactly
#: comes first: "Diamond Truss Lattice" must not be whittled down to "diamond".
_DISPLAY_KEYS: dict[str, str] = {
    _flatten(family.display_name): family.key for family in _FAMILY_LIST
}


def normalize_family_key(value: str) -> str:
    """Return the canonical family key for a display name or key.

    Accepts the inspector's display strings, the stored graph-property values
    from older projects, and the keys themselves. Returns ``"solid"`` for the
    solid envelope and ``""`` when nothing matches.
    """
    text = _flatten(value)
    if text in _DISPLAY_KEYS:
        return _DISPLAY_KEYS[text]
    compact = text.replace(" ", "_")
    if compact in FAMILIES:
        return compact
    if text in _ALIASES:
        return _ALIASES[text]
    # Only now fall back to trimming a trailing noun, and only one of them, so
    # a two-word family name keeps both of its words.
    for suffix in (" lattice", " truss", " network", " (skeletal)", " skeletal"):
        if text.endswith(suffix):
            trimmed = text[: -len(suffix)].strip()
            if trimmed in _DISPLAY_KEYS:
                return _DISPLAY_KEYS[trimmed]
            if trimmed.replace(" ", "_") in FAMILIES:
                return trimmed.replace(" ", "_")
            if trimmed in _ALIASES:
                return _ALIASES[trimmed]
            break
    return ""


def family_for(value: str) -> LatticeFamily | None:
    """Return the registry entry for a display name or key, or ``None``."""
    return FAMILIES.get(normalize_family_key(value))


# ── field evaluation ──────────────────────────────────────────────────────────


def _voxel_axes(
    shape: tuple[int, int, int],
    cell_size: float,
    origin: tuple[int, int, int],
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return broadcastable 1-D cell coordinates for voxel-centred sampling."""
    pitch = max(float(cell_size), 1e-9)
    axes = []
    for axis, count in enumerate(shape):
        index = np.arange(int(count), dtype=float) + float(origin[axis]) + 0.5
        value = scale * index / pitch
        axes.append(value.reshape([-1 if i == axis else 1 for i in range(3)]))
    return tuple(axes)  # type: ignore[return-value]


def tpms_implicit_field(
    shape: tuple[int, int, int],
    family_key: str,
    cell_size: float,
    *,
    origin: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Evaluate a TPMS level-set function over a voxel grid.

    Separable by construction: the trigonometric terms are evaluated on three
    1-D axes and only the products are formed at full size. On a 400^3 grid
    that is the difference between three 1.5 GB coordinate arrays plus nine
    full-size transcendental evaluations, and a handful of broadcasts.

    ``origin`` is the index of the block's first voxel in the full grid, so a
    caller may evaluate only the bounding box of the design envelope and still
    get a field in phase with the rest of the part.
    """
    family = FAMILIES.get(family_key)
    surface = _SURFACES.get(family.base if family is not None else family_key)
    if surface is None:
        raise ValueError(f"{family_key!r} is not a TPMS family.")
    dims = tuple(int(v) for v in shape)
    x, y, z = _voxel_axes(dims, cell_size, origin, 2.0 * np.pi)
    return np.broadcast_to(surface(x, y, z), dims).astype(float, copy=True)


def tpms_unit_cell_field(family_key: str, samples: int) -> np.ndarray:
    """Evaluate one periodic cell of a TPMS family on an ``samples^3`` grid."""
    count = int(samples)
    return tpms_implicit_field((count, count, count), family_key, float(count))


_SPECIFIC_AREA_CACHE: dict[str, float] = {}

#: Fallback specific area, used only if the measurement below cannot run. It is
#: the gyroid's value, so a family that falls back is merely mis-sized rather
#: than wildly so.
_DEFAULT_SPECIFIC_AREA = 3.09


def tpms_specific_surface_area(family_key: str) -> float:
    """Area of one periodic cell's mid-surface, per unit cell volume.

    This is the constant that converts a wall thickness into a relative
    density: a sheet of thickness ``t`` wrapped around a surface of area ``A``
    per unit cell occupies ``rho = A * t / a`` of that cell, in the thin-wall
    limit where the two offset faces have not yet met at the necks.

    It is measured from the same nodal field the part is rasterized from rather
    than quoted from a table, for the reason the module header gives: a second
    copy of the cell geometry is a second chance for the manufactured part to
    stop matching the surface the optimizer was driven by. The measured values
    do agree with the published ones — gyroid 3.09, Schwarz D 3.84, Schwarz P
    2.35 — which is the check that the nodal approximations here are the
    surfaces they claim to be.

    Assuming one shared constant instead is what previously let a stated
    printable wall mean different things per family: at ``t/a = 1/8`` the old
    fixed factor of 2 claimed 25% relative density for every surface, where the
    gyroid really reaches 39% and the Schwarz P 29%.
    """
    key = str(family_key)
    cached = _SPECIFIC_AREA_CACHE.get(key)
    if cached is not None:
        return cached
    area = _DEFAULT_SPECIFIC_AREA
    try:
        from skimage.measure import marching_cubes, mesh_surface_area

        # One cell, padded periodically by one sample either side so the
        # extraction closes across the cell boundary instead of capping there.
        samples = 48
        field = tpms_implicit_field(
            (samples + 2,) * 3, key, float(samples), origin=(-1, -1, -1)
        )
        vertices, faces, _, _ = marching_cubes(field, level=0.0)
        # Trim to the faces whose centroid lies inside the unpadded cell, so
        # the padding contributes closure but not area.
        centroids = vertices[faces].mean(axis=1)
        inside = np.all((centroids >= 1.0) & (centroids <= samples + 1.0), axis=1)
        if np.any(inside):
            area = float(
                mesh_surface_area(vertices, faces[inside])
            ) / float(samples**2)
    except Exception:  # pragma: no cover - measurement is best effort
        logger = __import__("logging").getLogger(__name__)
        logger.debug("Specific area measurement failed for %s", key, exc_info=True)
    area = float(np.clip(area, 1.0, 6.0))
    _SPECIFIC_AREA_CACHE[key] = area
    return area


def strut_edges(family_key: str) -> np.ndarray:
    """Return the ``(n, 2, 3)`` unit-cell edge list of a strut family."""
    family = FAMILIES.get(family_key)
    key = family.base if family is not None else str(family_key)
    edges = _STRUT_EDGES.get(key)
    if edges is None:
        raise ValueError(f"{family_key!r} is not a strut family.")
    return edges


#: Largest member radius, as a fraction of the cell pitch, that the rasterizer
#: ever has to resolve. Half the pitch is where opposing members merge and the
#: cell stops being a lattice, so a centreline further than this from the cell
#: can never put material inside it and is not worth a distance evaluation.
MAXIMUM_MEMBER_RADIUS = 0.5

_TILED_EDGE_CACHE: dict[tuple[str, int], np.ndarray] = {}


def _segment_distance_to_unit_cell(start: np.ndarray, end: np.ndarray) -> float:
    """Shortest distance from a segment to the closed box ``[0, 1]^3``."""
    parameters = np.linspace(0.0, 1.0, 65)[:, None]
    points = start[None, :] * (1.0 - parameters) + end[None, :] * parameters
    outside = np.clip(points, 0.0, 1.0) - points
    return float(np.min(np.linalg.norm(outside, axis=1)))


def _tiled_edges(
    family_key: str,
    margin: float = MAXIMUM_MEMBER_RADIUS,
) -> np.ndarray:
    """Replicate the unit-cell edges over the neighbouring cells.

    A point just inside one face of the cell can have its nearest strut in the
    neighbouring cell, and the periodic wrap does not bring that strut into the
    cell's own edge list. The diamond truss is the clear case: no bond in the
    base list touches the (1,1,1) corner at all, because in the diamond-cubic
    graph that site's four bonds all leave the cell.

    Replicating over the 27 neighbours, discarding duplicates, and dropping
    segments that cannot come within ``margin`` of the cell leaves the smallest
    set that still gives an exact distance field wherever the field can put
    material. ``margin`` is the largest radius plus raster allowance the caller
    will threshold at: beyond it the distance is allowed to be an overestimate,
    because no voxel there is solid either way. Passing the real working margin
    rather than the 0.5 ceiling is worth roughly a factor of three in segments.
    """
    margin = float(np.clip(margin, 0.02, MAXIMUM_MEMBER_RADIUS))
    # Bucket the margin so a continuously varying request still hits the cache,
    # always rounding up so the cached set is never short of segments.
    bucket = int(np.ceil(margin * 40.0))
    cache_key = (family_key, bucket)
    cached = _TILED_EDGE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    margin = bucket / 40.0

    base = strut_edges(family_key)
    offsets = np.asarray(
        [
            (i, j, k)
            for i in (-1.0, 0.0, 1.0)
            for j in (-1.0, 0.0, 1.0)
            for k in (-1.0, 0.0, 1.0)
        ],
        dtype=float,
    )
    tiled = (base[None, :, :, :] + offsets[:, None, None, :]).reshape((-1, 2, 3))

    kept: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    for segment in tiled:
        # An undirected segment: order the ends so the two directions of the
        # same strut are recognized as one.
        ends = sorted(
            tuple(int(round(float(v) * 1_000_000)) for v in point)
            for point in segment
        )
        key = tuple(ends[0]) + tuple(ends[1])
        if key in seen:
            continue
        seen.add(key)
        if _segment_distance_to_unit_cell(segment[0], segment[1]) > margin:
            continue
        kept.append(segment)

    result = np.asarray(kept, dtype=float).reshape((-1, 2, 3))
    _TILED_EDGE_CACHE[cache_key] = result
    return result


#: Sub-voxel raster allowance, in voxels, for the two kinds of strut. A member
#: whose centreline lies on a cell boundary otherwise misses every voxel centre
#: and a nominally one-voxel strut disappears. An axis-aligned member is only
#: offset within the plane normal to it, so its allowance is the in-plane voxel
#: half-diagonal; a general member can be offset in three dimensions and 0.5 is
#: the value the octet family was calibrated on. Splitting them matters for the
#: mixed families (BCC-Z, FCC-Z), which contain both kinds at once.
RASTER_ALLOWANCE_AXIAL = float(np.sqrt(2.0) * 0.5)
RASTER_ALLOWANCE_DIAGONAL = 0.5


def strut_distance_field(
    shape: tuple[int, int, int],
    family_key: str,
    cell_size: float,
    *,
    origin: tuple[int, int, int] = (0, 0, 0),
    margin: float = MAXIMUM_MEMBER_RADIUS,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Distance from every voxel to the nearest strut centreline.

    Returns a ``(2, *shape)`` stack in units of the cell pitch: index 0 is the
    distance to the axis-aligned members, index 1 the distance to everything
    else, each ``inf`` where the family has no member of that kind. They are
    kept apart so the caller can apply the right raster allowance to each, which
    is what makes the mixed families (BCC-Z, FCC-Z) come out right.

    With ``mask``, only the selected voxels are evaluated and the result is
    ``(2, count)`` in ``np.argwhere`` order. A lattice only exists inside the
    optimized envelope, which is usually a quarter of its bounding box, so
    evaluating the rest is pure waste — and the segment loop is the dominant
    cost of building a strut lattice.

    Beyond ``margin`` the returned distance is only an upper bound; see
    :func:`_tiled_edges`. Thresholding this against a radius — constant, or
    graded from the optimizer's density — is the whole rasterization.
    """
    dims = tuple(int(v) for v in shape)
    if mask is None:
        axes = _voxel_axes(dims, cell_size, origin, 1.0)
        unit = tuple(np.mod(axis, 1.0) for axis in axes)
        out_shape: tuple[int, ...] = dims
    else:
        selected = np.asarray(mask, dtype=bool)
        if selected.shape != dims:
            raise ValueError("Strut distance mask must match the block shape.")
        indices = np.nonzero(selected)
        pitch = max(float(cell_size), 1e-9)
        unit = tuple(
            np.mod(
                (indices[axis].astype(float) + float(origin[axis]) + 0.5) / pitch,
                1.0,
            )
            for axis in range(3)
        )
        out_shape = (int(indices[0].size),)

    edges = _tiled_edges(family_key, margin)
    best = np.full((2,) + out_shape, np.inf, dtype=float)
    for start, end in edges:
        segment = end - start
        length_sq = float(np.dot(segment, segment))
        if length_sq <= 1e-18:
            continue
        target = best[
            0 if int(np.count_nonzero(np.abs(segment) > 1e-12)) == 1 else 1
        ]
        # Broadcast the projection out of the three 1-D axes rather than
        # materializing a (*shape, 3) coordinate array; on a recovery grid that
        # array is the single largest allocation in the whole build.
        projection = (
            (unit[0] - start[0]) * segment[0]
            + (unit[1] - start[1]) * segment[1]
            + (unit[2] - start[2]) * segment[2]
        ) / length_sq
        projection = np.clip(projection, 0.0, 1.0)
        distance_sq = np.zeros(out_shape, dtype=float)
        for axis in range(3):
            delta = (unit[axis] - start[axis]) - projection * segment[axis]
            distance_sq = distance_sq + delta * delta
        np.minimum(target, distance_sq, out=target)
    return np.sqrt(best)


def strut_unit_cell_distance(family_key: str, samples: int) -> np.ndarray:
    """Distance stack over exactly one periodic cell, on ``samples^3`` voxels."""
    count = max(2, int(samples))
    return strut_distance_field(
        (count, count, count), family_key, float(count)
    )


def strut_occupancy(
    distance: np.ndarray,
    radius: np.ndarray | float,
    voxels_per_cell: float,
) -> np.ndarray:
    """Threshold a strut distance stack into a solid mask.

    ``radius`` is in units of the cell pitch and may be an array, which is how
    a de-homogenized lattice varies its member thickness with the optimizer's
    local density.

    The raster allowance is a *floor* on the thresholding radius, not a term
    added to it. Both forms keep a sub-voxel member alive, which is what the
    allowance is for — a centreline lying between voxel centres otherwise
    misses every sample and the member disappears. Only the floor form stops
    there: adding the allowance instead thickens every member by the same
    ~0.5-0.7 voxels whether it needed it or not, and that is not a rounding
    detail at the radii a lattice actually uses. On a 16-voxel cell a 10%
    relative-density strut has a radius near 0.8 voxels, so a 0.7 voxel
    addition nearly doubles it: measured across the strut families, a
    requested 10% cell was built at 15-22%. The floor form leaves a
    well-resolved member exactly where the density asked for it.
    """
    pitch = max(float(voxels_per_cell), 1e-9)
    axial = distance[0] <= np.maximum(radius, RASTER_ALLOWANCE_AXIAL / pitch)
    diagonal = distance[1] <= np.maximum(
        radius, RASTER_ALLOWANCE_DIAGONAL / pitch
    )
    return axial | diagonal


def strut_raster_margin(
    maximum_radius: float,
    voxels_per_cell: float,
) -> float:
    """Largest distance a threshold can reach, for pruning the segment set."""
    return float(maximum_radius) + RASTER_ALLOWANCE_AXIAL / max(
        float(voxels_per_cell), 1e-9
    )


def strut_network(
    family_key: str,
    cell_counts: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Tile a strut family into node coordinates and node-index members.

    Coordinates are fractional over the whole tiled block, so the result maps
    onto a voxel grid or onto model units with one scale. Coincident nodes are
    merged, which is what makes the network a connected truss rather than a
    pile of disjoint segments — the truss sizing and the analytic CAD path both
    depend on it.
    """
    edges = strut_edges(family_key)
    counts = np.maximum(1, np.asarray(cell_counts, dtype=int))
    span = counts.astype(float)

    nodes: list[np.ndarray] = []
    node_ids: dict[tuple[int, int, int], int] = {}
    members: set[tuple[int, int]] = set()

    def node_index(point: np.ndarray) -> int:
        # Quantize before hashing: the same corner reached from two cells must
        # be one node or the truss silently comes apart at every cell face.
        key = tuple(int(round(float(value) * 1_000_000)) for value in point)
        index = node_ids.get(key)
        if index is None:
            index = len(nodes)
            node_ids[key] = index
            nodes.append(np.asarray(point, dtype=float))
        return index

    for ix in range(int(counts[0])):
        for iy in range(int(counts[1])):
            for iz in range(int(counts[2])):
                offset = np.asarray((ix, iy, iz), dtype=float)
                for start, end in edges:
                    a = node_index((start + offset) / span)
                    b = node_index((end + offset) / span)
                    if a != b:
                        members.add((min(a, b), max(a, b)))

    return (
        np.asarray(nodes, dtype=float).reshape((-1, 3)),
        np.asarray(sorted(members), dtype=int).reshape((-1, 2)),
    )
