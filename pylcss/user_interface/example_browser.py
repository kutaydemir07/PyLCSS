# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Discoverable, engineering-focused browser for bundled PyLCSS examples."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PySide6 import QtCore, QtWidgets


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CAD_CATEGORY_NAMES = {
    "01_fea": "Engineering Challenge - Structural",
    "02_impact": "Engineering Challenge - Impact",
    "topology_optimization": "Engineering Challenge - Topology",
    "lattice_optimization": "Engineering Challenge - Lattice",
}
# The browser lists categories in discovery order, and a directory name that
# sorts well on disk is not the order an engineer works in. A lattice study is
# a topology study whose result is a graded cell rather than a solid, so it
# follows topology instead of preceding it alphabetically.
_CAD_CATEGORY_ORDER = (
    "00_modeling",
    "01_fea",
    "02_impact",
    "topology_optimization",
    "lattice_optimization",
)


@dataclass(frozen=True, slots=True)
class RequirementSummary:
    """One project requirement shown before the challenge is opened."""

    requirement: str
    target: str
    verification: str
    baseline_status: str


@dataclass(frozen=True, slots=True)
class ExampleSummary:
    """Metadata shown before a bundled example is opened."""

    path: Path
    title: str
    category: str
    purpose: str
    fidelity: str
    visual_target: str
    engineering_decision: str
    baseline: str
    design_variables: tuple[str, ...]
    load_cases: tuple[str, ...]
    requirements: tuple[RequirementSummary, ...]
    deliverables: tuple[str, ...]
    workflow: tuple[str, ...]
    acceptance_criteria: tuple[str, ...]
    capabilities: tuple[str, ...]
    node_count: int
    connection_count: int


def _text_list(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if not isinstance(value, Iterable) or isinstance(value, (dict, bytes)):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _requirements(value: Any) -> tuple[RequirementSummary, ...]:
    if not isinstance(value, list):
        return ()
    requirements: list[RequirementSummary] = []
    for item in value:
        if isinstance(item, str):
            requirement = item.strip()
            if requirement:
                requirements.append(
                    RequirementSummary(requirement, "", "", "Open")
                )
            continue
        if not isinstance(item, dict):
            continue
        requirement = str(
            item.get("requirement") or item.get("name") or ""
        ).strip()
        if not requirement:
            continue
        requirements.append(
            RequirementSummary(
                requirement=requirement,
                target=str(item.get("target") or "").strip(),
                verification=str(item.get("verification") or "").strip(),
                baseline_status=str(
                    item.get("baseline_status") or "Open"
                ).strip(),
            )
        )
    return tuple(requirements)


def _cad_capabilities(nodes: dict[str, Any]) -> tuple[str, ...]:
    types = {str(item.get("type_") or "") for item in nodes.values()}
    labels: list[str] = []
    checks = (
        ("com.cad.sim.solver", "Static FEA"),
        ("com.cad.sim.component", "Multi-body / multi-material"),
        ("com.cad.sim.pressure_load", "Pressure loading"),
        ("com.cad.sim.crash_solver", "Explicit impact"),
        ("com.cad.sim.topopt_voxel", "Topology optimization"),
        ("com.cad.sim.lattice_voxel", "Lattice optimization"),
        ("com.cad.assembly", "Multiple CAD bodies"),
        ("com.cad.code_part", "Parametric code geometry"),
    )
    for prefix, label in checks:
        if any(node_type == prefix or node_type.startswith(prefix + ".") for node_type in types):
            labels.append(label)
    return tuple(labels or ["Parametric modeling"])


def _cad_category_sort_key(path: Path) -> tuple[int, str, str]:
    """Order examples by workflow stage, then by file name inside a stage."""
    directory = path.parent.name
    try:
        rank = _CAD_CATEGORY_ORDER.index(directory)
    except ValueError:
        rank = len(_CAD_CATEGORY_ORDER)
    return (rank, directory, path.name)


def _load_cad_examples() -> list[ExampleSummary]:
    root = _REPO_ROOT / "data" / "cad_environment"
    examples: list[ExampleSummary] = []
    for path in sorted(root.rglob("*.cad"), key=_cad_category_sort_key):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        nodes = data.get("nodes") if isinstance(data.get("nodes"), dict) else {}
        connections = data.get("connections")
        category = _CAD_CATEGORY_NAMES.get(
            path.parent.name,
            path.parent.name.replace("_", " ").title(),
        )
        examples.append(
            ExampleSummary(
                path=path,
                title=str(data.get("_title") or path.stem.replace("_", " ").title()),
                category=category,
                purpose=str(data.get("_purpose") or "Bundled Design Studio workflow."),
                fidelity=str(
                    data.get("_fidelity")
                    or "Review the model assumptions and verify convergence before design use."
                ),
                visual_target=str(
                    data.get("_visual_target")
                    or "Review the final geometry and reported engineering fields."
                ),
                engineering_decision=str(
                    data.get("_engineering_decision")
                    or "Decide whether the design satisfies the stated requirements."
                ),
                baseline=str(
                    data.get("_baseline")
                    or "Run the supplied baseline before changing the design."
                ),
                design_variables=_text_list(data.get("_design_variables")),
                load_cases=_text_list(data.get("_load_cases")),
                requirements=_requirements(data.get("_requirements")),
                deliverables=_text_list(data.get("_deliverables")),
                workflow=_text_list(data.get("_workflow")),
                acceptance_criteria=_text_list(data.get("_acceptance_criteria")),
                capabilities=_cad_capabilities(nodes),
                node_count=len(nodes),
                connection_count=(
                    len(connections) if isinstance(connections, list) else 0
                ),
            )
        )
    return examples


def _load_modeling_examples() -> list[ExampleSummary]:
    root = _REPO_ROOT / "data" / "modeling_environment"
    examples: list[ExampleSummary] = []
    for path in sorted(root.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        systems = data.get("systems") if isinstance(data.get("systems"), list) else []
        node_count = 0
        connection_count = 0
        for system in systems:
            graph = system.get("graph") if isinstance(system, dict) else {}
            nodes = graph.get("nodes") if isinstance(graph, dict) else {}
            connections = graph.get("connections") if isinstance(graph, dict) else []
            node_count += len(nodes) if isinstance(nodes, dict) else 0
            connection_count += len(connections) if isinstance(connections, list) else 0
        examples.append(
            ExampleSummary(
                path=path,
                title=str(
                    data.get("_title")
                    or data.get("product_name")
                    or path.stem
                ),
                category="System Modeling",
                purpose=str(
                    data.get("_purpose")
                    or "Multidisciplinary system model for solution-space exploration."
                ),
                fidelity=str(
                    data.get("_fidelity")
                    or "Equation-level demonstration; validate discipline models and units "
                    "before product decisions."
                ),
                visual_target=str(
                    data.get("_visual_target")
                    or "Review the resulting system responses and feasible design space."
                ),
                engineering_decision=str(
                    data.get("_engineering_decision") or ""
                ),
                baseline=str(data.get("_baseline") or ""),
                design_variables=_text_list(data.get("_design_variables")),
                load_cases=_text_list(data.get("_load_cases")),
                requirements=_requirements(data.get("_requirements")),
                deliverables=_text_list(data.get("_deliverables")),
                workflow=_text_list(data.get("_workflow")),
                acceptance_criteria=_text_list(data.get("_acceptance_criteria")),
                capabilities=(
                    f"{len(systems)} discipline system"
                    + ("s" if len(systems) != 1 else ""),
                    "Solution-space ready",
                ),
                node_count=node_count,
                connection_count=connection_count,
            )
        )
    return examples


def bundled_examples(kind: str) -> list[ExampleSummary]:
    """Return bundled examples for ``design`` or ``modeling``."""
    return _load_cad_examples() if kind == "design" else _load_modeling_examples()


class ExampleBrowserDialog(QtWidgets.QDialog):
    """Select an example while seeing its intent, scope, and review gates."""

    def __init__(self, kind: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._examples = bundled_examples(kind)
        self._selected_path: Path | None = None
        self.setWindowTitle(
            "Engineering Challenges" if kind == "design" else "Examples"
        )
        self.resize(1080, 650)

        outer = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "Choose a decision-driven engineering challenge. Run the baseline, "
            "change the design, and verify every requirement."
            if kind == "design"
            else "Choose an example."
        )
        intro.setWordWrap(True)
        outer.addWidget(intro)

        search = QtWidgets.QLineEdit()
        search.setPlaceholderText("Search examples")
        search.setClearButtonEnabled(True)
        outer.addWidget(search)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self._tree = QtWidgets.QTreeWidget()
        self._tree.setHeaderLabels(["Example", "Workflow"])
        self._tree.setRootIsDecorated(True)
        self._tree.setAlternatingRowColors(True)
        self._tree.setColumnWidth(0, 335)
        self._tree.setColumnWidth(1, 145)
        splitter.addWidget(self._tree)

        detail_scroll = QtWidgets.QScrollArea()
        detail_scroll.setWidgetResizable(True)
        detail = QtWidgets.QWidget()
        detail_layout = QtWidgets.QVBoxLayout(detail)
        self._title = QtWidgets.QLabel()
        self._title.setStyleSheet("font-size:17px; font-weight:700;")
        self._title.setWordWrap(True)
        detail_layout.addWidget(self._title)
        self._capabilities = QtWidgets.QLabel()
        self._capabilities.setWordWrap(True)
        self._capabilities.setStyleSheet("color:#64B5F6; font-weight:600;")
        detail_layout.addWidget(self._capabilities)
        self._details = QtWidgets.QTextBrowser()
        self._details.setOpenExternalLinks(False)
        detail_layout.addWidget(self._details, 1)
        detail_scroll.setWidget(detail)
        splitter.addWidget(detail_scroll)
        splitter.setSizes([480, 600])
        outer.addWidget(splitter, 1)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Open | QtWidgets.QDialogButtonBox.Cancel
        )
        self._open_button = buttons.button(QtWidgets.QDialogButtonBox.Open)
        self._open_button.setText("Open")
        self._open_button.setEnabled(False)
        buttons.accepted.connect(self._accept_selection)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

        self._populate()
        self._tree.currentItemChanged.connect(self._show_item)
        self._tree.itemDoubleClicked.connect(lambda *_args: self._accept_selection())
        search.textChanged.connect(self._filter)
        if self._tree.topLevelItemCount():
            first_category = self._tree.topLevelItem(0)
            first_category.setExpanded(True)
            if first_category.childCount():
                self._tree.setCurrentItem(first_category.child(0))

    @property
    def selected_path(self) -> Path | None:
        return self._selected_path

    def _populate(self) -> None:
        categories: dict[str, QtWidgets.QTreeWidgetItem] = {}
        for index, example in enumerate(self._examples):
            category = categories.get(example.category)
            if category is None:
                category = QtWidgets.QTreeWidgetItem([example.category, ""])
                category.setFirstColumnSpanned(True)
                font = category.font(0)
                font.setBold(True)
                category.setFont(0, font)
                categories[example.category] = category
                self._tree.addTopLevelItem(category)
            child = QtWidgets.QTreeWidgetItem(
                [example.title, ", ".join(example.capabilities)]
            )
            child.setData(0, QtCore.Qt.UserRole, index)
            category.addChild(child)

    def _show_item(
        self,
        current: QtWidgets.QTreeWidgetItem | None,
        _previous: QtWidgets.QTreeWidgetItem | None,
    ) -> None:
        index = current.data(0, QtCore.Qt.UserRole) if current is not None else None
        if not isinstance(index, int):
            self._open_button.setEnabled(False)
            return
        example = self._examples[index]
        self._selected_path = example.path
        self._open_button.setEnabled(True)
        self._title.setText(example.title)
        self._capabilities.setText("  •  ".join(example.capabilities))
        esc = lambda value: html.escape(str(value))
        workflow = "".join(f"<li>{esc(item)}</li>" for item in example.workflow)
        criteria = "".join(
            f"<li>{esc(item)}</li>" for item in example.acceptance_criteria
        )
        design_variables = "".join(
            f"<li>{esc(item)}</li>" for item in example.design_variables
        )
        load_cases = "".join(
            f"<li>{esc(item)}</li>" for item in example.load_cases
        )
        deliverables = "".join(
            f"<li>{esc(item)}</li>" for item in example.deliverables
        )
        requirement_rows = "".join(
            "<tr>"
            f"<td>{esc(item.requirement)}</td>"
            f"<td>{esc(item.target)}</td>"
            f"<td>{esc(item.verification)}</td>"
            f"<td><b>{esc(item.baseline_status)}</b></td>"
            "</tr>"
            for item in example.requirements
        )
        self._details.setHtml(
            "<h3>Decision to make</h3>"
            f"<p>{esc(example.engineering_decision)}</p>"
            "<h3>Problem</h3>"
            f"<p>{esc(example.purpose)}</p>"
            "<h3>Supplied baseline</h3>"
            f"<p>{esc(example.baseline)}</p>"
            + (
                f"<h3>Design variables</h3><ul>{design_variables}</ul>"
                if design_variables
                else ""
            )
            + (
                f"<h3>Load cases</h3><ul>{load_cases}</ul>"
                if load_cases
                else ""
            )
            + (
                "<h3>Project requirements</h3>"
                "<table border='1' cellspacing='0' cellpadding='4'>"
                "<tr><th>Requirement</th><th>Target</th>"
                "<th>Verification</th><th>Baseline</th></tr>"
                f"{requirement_rows}</table>"
                if requirement_rows
                else ""
            )
            + "<h3>Model fidelity and limits</h3>"
            f"<p>{esc(example.fidelity)}</p>"
            "<h3>Expected result</h3>"
            f"<p>{esc(example.visual_target)}</p>"
            + (f"<h3>Workflow</h3><ol>{workflow}</ol>" if workflow else "")
            + (
                f"<h3>Acceptance criteria</h3><ul>{criteria}</ul>"
                if criteria
                else ""
            )
            + (
                f"<h3>Required deliverables</h3><ul>{deliverables}</ul>"
                if deliverables
                else ""
            )
            + "<h3>Graph size</h3>"
            f"<p>{example.node_count} nodes, "
            f"{example.connection_count} connections</p>"
            "<h3>Source</h3>"
            f"<p>{example.path.relative_to(_REPO_ROOT)}</p>"
        )

    def _filter(self, text: str) -> None:
        needle = text.strip().casefold()
        for category_index in range(self._tree.topLevelItemCount()):
            category = self._tree.topLevelItem(category_index)
            visible_children = 0
            for child_index in range(category.childCount()):
                child = category.child(child_index)
                index = child.data(0, QtCore.Qt.UserRole)
                example = self._examples[index]
                haystack = " ".join(
                    (
                        example.title,
                        example.category,
                        example.purpose,
                        example.fidelity,
                        example.visual_target,
                        example.engineering_decision,
                        example.baseline,
                        *example.capabilities,
                        *example.design_variables,
                        *example.load_cases,
                        *(
                            field
                            for requirement in example.requirements
                            for field in (
                                requirement.requirement,
                                requirement.target,
                                requirement.verification,
                                requirement.baseline_status,
                            )
                        ),
                        *example.deliverables,
                        *example.workflow,
                        *example.acceptance_criteria,
                    )
                ).casefold()
                visible = not needle or needle in haystack
                child.setHidden(not visible)
                visible_children += int(visible)
            category.setHidden(visible_children == 0)
            if needle and visible_children:
                category.setExpanded(True)

    def _accept_selection(self) -> None:
        if self._selected_path is not None:
            self.accept()


def choose_bundled_example(
    kind: str,
    parent: QtWidgets.QWidget | None = None,
) -> Path | None:
    """Open the browser and return the chosen file path."""
    dialog = ExampleBrowserDialog(kind, parent)
    return dialog.selected_path if dialog.exec() else None


__all__ = [
    "ExampleBrowserDialog",
    "ExampleSummary",
    "RequirementSummary",
    "bundled_examples",
    "choose_bundled_example",
]
