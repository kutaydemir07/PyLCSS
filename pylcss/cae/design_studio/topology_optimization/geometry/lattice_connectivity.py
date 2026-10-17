"""Conservative material connectivity for continuous, clipped lattice fields."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


class LatticeConnectivityError(ValueError):
    """Prescribed material cannot be retained as a connected lattice body."""

    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def filter_anchored_lattice_material(level, envelope, solid_level, void_level, *,
                                    threshold, spacing, progress_callback=None):
    """Remove only material fragments without a prescribed-solid attachment.

    Six-neighbour adjacency excludes mere vertex/edge contacts. Prescribed
    anchors must share one material component within each original envelope
    body. Separate source bodies remain separate. Without anchors all material
    is retained: neither size nor a surface-shell count identifies its intent.
    The signed field is changed only at removed positive material samples;
    negative void samples and hence existing cavities are never filled.
    """
    arrays = [np.asarray(value) for value in (level, envelope, solid_level, void_level)]
    if len({value.shape for value in arrays}) != 1 or arrays[0].ndim != 3:
        raise ValueError("Lattice material and prescribed-region fields must share a 3-D grid")
    level, envelope, solid_level, void_level = arrays
    threshold = float(threshold)
    spacing = np.asarray(spacing, dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0.):
        raise ValueError("Lattice connectivity needs positive physical grid spacing")
    if callable(progress_callback):
        progress_callback("Checking lattice material attachments")
    labels, count = ndi.label(level > threshold)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    total = int(counts.sum())
    prescribed = (solid_level > threshold) & (envelope > threshold) & (void_level < -threshold)
    prescribed_count = int(np.count_nonzero(prescribed))
    report = {
        "method": "Six-neighbour positive material rooted in prescribed solid regions",
        "adjacency": "shared grid face; point and edge contacts do not connect material",
        "contour_threshold": threshold,
        "material_components_before": int(count), "material_components_after": int(count),
        "prescribed_sample_count": prescribed_count,
        "removed_material_sample_count": 0, "removed_material_volume": 0.,
        "removed_material_fraction": 0.,
        "all_prescribed_regions_connected": None,
    }
    if not prescribed_count:
        report.update(cleanup_applied=False, status="No prescribed anchors; all input material components retained")
        return report
    missing = prescribed & (labels == 0)
    if np.any(missing):
        report.update(status="Prescribed material missing", missing_prescribed_sample_count=int(missing.sum()),
                      all_prescribed_regions_connected=False)
        raise LatticeConnectivityError("Lattice generation would remove prescribed solid material. "
            "Check overlapping void regions and increase sampling quality.", report)
    # Original-envelope components distinguish intentional disjoint bodies
    # from a broken lattice inside one body. This is not a largest-body rule.
    domain_labels, domain_count = ndi.label((envelope > threshold) & (void_level < -threshold))
    pairs = set()
    for start in range(0, level.shape[0], 16):
        if callable(progress_callback):
            progress_callback("Checking prescribed lattice regions")
        selection = slice(start, min(start + 16, level.shape[0]))
        mask = prescribed[selection]
        local_pairs = np.unique(np.column_stack((domain_labels[selection][mask], labels[selection][mask])), axis=0)
        pairs.update(map(tuple, local_pairs.tolist()))
    anchor_pairs = np.asarray(sorted(pairs), dtype=np.int32)
    groups = []
    for domain in np.unique(anchor_pairs[:, 0]):
        materials = anchor_pairs[anchor_pairs[:, 0] == domain, 1]
        groups.append({"envelope_body": int(domain),
                       "anchored_material_components": [int(value) for value in materials]})
        if len(materials) != 1:
            report.update(status="Prescribed regions are disconnected", anchor_groups=groups,
                          original_envelope_components=int(domain_count),
                          all_prescribed_regions_connected=False)
            raise LatticeConnectivityError(
                "The lattice does not connect all prescribed solid regions within one source body. "
                "Use a smaller cell size, a thicker member, or an explicit boundary skin; "
                "no mounting region was discarded and no connecting ligament was invented.", report)
    keep_ids = np.unique(anchor_pairs[:, 1])
    keep = np.zeros(len(counts), dtype=bool)
    keep[keep_ids] = True
    # Unanchored original bodies must not be silently erased when another
    # independent source body happens to contain anchors.
    anchored_domains = np.unique(anchor_pairs[:, 0])
    preserve_domain = np.ones(domain_count + 1, dtype=bool)
    preserve_domain[0] = False
    preserve_domain[anchored_domains] = False
    for start in range(0, level.shape[0], 16):
        selection = slice(start, min(start + 16, level.shape[0]))
        extra_ids = np.unique(labels[selection][preserve_domain[domain_labels[selection]]])
        keep[extra_ids] = True
    keep[0] = False
    remove_ids = np.flatnonzero((counts > 0) & ~keep)
    removed = int(counts[remove_ids].sum())
    if removed:
        if callable(progress_callback):
            progress_callback("Removing unattached lattice clipping fragments")
        remove_lookup = np.zeros(len(counts), dtype=bool)
        remove_lookup[remove_ids] = True
        removed_mask = remove_lookup[labels]
        level[removed_mask] = -np.maximum(np.abs(level[removed_mask]),
                                         max(float(np.min(spacing)) * 1e-6, abs(threshold)))
    # Check prescribed samples after the only permitted field mutation.
    if np.any(level[prescribed] <= threshold):
        raise RuntimeError("Lattice connectivity cleanup changed prescribed material")
    report.update(cleanup_applied=bool(removed), status="Prescribed attachments connected",
                  original_envelope_components=int(domain_count), anchor_groups=groups,
                  material_components_after=int(np.count_nonzero((counts > 0) & keep)),
                  removed_material_component_count=int(len(remove_ids)),
                  removed_material_sample_count=removed,
                  removed_material_volume=float(removed * np.prod(spacing)),
                  removed_material_fraction=removed / max(total, 1),
                  all_prescribed_regions_connected=True,
                  unanchored_source_bodies_retained=int(np.count_nonzero(preserve_domain)))
    return report


__all__ = ["filter_anchored_lattice_material", "LatticeConnectivityError"]
