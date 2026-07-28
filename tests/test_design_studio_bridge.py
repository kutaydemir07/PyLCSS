import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtWidgets

from pylcss.system_modeling.design_studio_bridge import (
    SimulationFunctionSpec,
    SimulationInputSpec,
    SimulationOutputSpec,
    find_input,
    generate_simulation_function_code,
    inspect_design_studio_study,
    make_default_spec,
)
from pylcss.user_interface.system_modeling.design_studio_bridge_dialog import (
    DesignStudioBridgeDialog,
)
from pylcss.user_interface.system_modeling.system_modeling_widget import (
    ModelingWidget,
)
from pylcss.user_interface.system_modeling.actions import (
    load_graph_from_file,
    save_graph_to_file,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY_PATH = (
    REPO_ROOT
    / "data"
    / "cad_environment"
    / "01_fea"
    / "04_nonlinear_fea_benchmark_plate.cad"
)


@pytest.fixture(scope="module")
def qapp():
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield application
    application.processEvents()


def test_study_inspection_discovers_geometry_pressure_and_fea_outputs():
    descriptor = inspect_design_studio_study(STUDY_PATH)

    assert descriptor.title == "Nonlinear FEA Benchmark Plate"
    assert [analysis.kind for analysis in descriptor.analyses] == ["fea"]
    assert find_input(
        descriptor.inputs, source_kind="parameter", target="H"
    ).default == 10.0
    pressure = next(
        item
        for item in descriptor.inputs
        if item.source_kind == "setting" and item.target.endswith("::pressure")
    )
    assert pressure.default == 40.0
    assert pressure.unit == "MPa"
    assert {item.field for item in descriptor.outputs["fea"]} >= {
        "max_stress",
        "peak_disp",
        "mass",
    }


def test_generated_adapter_routes_parameters_settings_and_results():
    descriptor = inspect_design_studio_study(STUDY_PATH)
    thickness = find_input(
        descriptor.inputs, source_kind="parameter", target="H"
    )
    pressure = next(
        item
        for item in descriptor.inputs
        if item.source_kind == "setting" and item.target.endswith("::pressure")
    )
    spec = SimulationFunctionSpec(
        project_path=str(STUDY_PATH),
        analysis_kind="fea",
        node_name="Plate FEA",
        inputs=(
            SimulationInputSpec(
                "thickness",
                thickness.label,
                thickness.source_kind,
                thickness.target,
                thickness.default,
                thickness.lower,
                thickness.upper,
                thickness.unit,
            ),
            SimulationInputSpec(
                "pressure",
                pressure.label,
                pressure.source_kind,
                pressure.target,
                pressure.default,
                pressure.lower,
                pressure.upper,
                pressure.unit,
            ),
        ),
        outputs=(
            SimulationOutputSpec(
                "stress", "Maximum stress", "max_stress", "MPa"
            ),
        ),
    )
    statements = generate_simulation_function_code(spec)
    wrapped = "def adapter(thickness, pressure):\n" + "\n".join(
        f"    {line}" for line in statements.splitlines()
    ) + "\n    return stress\n"

    calls = []

    class FakeCad:
        @staticmethod
        def fea(path, _settings=None, **parameters):
            calls.append((path, _settings, parameters))
            return SimpleNamespace(max_stress=321.5)

    namespace = {"cad": FakeCad}
    exec(wrapped, namespace)
    assert namespace["adapter"](9.5, 2.2) == 321.5
    assert calls[0][2] == {"H": 9.5}
    assert calls[0][1][pressure.target] == 2.2


def test_bridge_dialog_defaults_to_geometry_and_key_fea_results(qapp):
    descriptor = inspect_design_studio_study(STUDY_PATH)
    dialog = DesignStudioBridgeDialog(descriptor, [], parent=None)
    dialog._accept_selection()
    spec = dialog.selected_spec()

    assert dialog.should_create_system() is True
    assert dialog.should_create_io_nodes() is True
    assert {item.port_name for item in spec.inputs} == {
        "L", "W", "H", "big_R", "bolt_x", "bolt_y", "bolt_d"
    }
    assert {item.result_field for item in spec.outputs} == {
        "max_stress", "peak_disp", "mass"
    }
    dialog.close()
    dialog.deleteLater()
    qapp.processEvents()


def test_modeling_widget_creates_wired_serializable_executable_graph(qapp):
    descriptor = inspect_design_studio_study(STUDY_PATH)
    spec = make_default_spec(descriptor)
    widget = ModelingWidget()
    node = widget.create_design_studio_function(
        spec,
        destination_system="Plate Study",
        create_system=True,
        create_io_nodes=True,
    )
    qapp.processEvents()

    assert node.type_.startswith("com.pfd.custom_block.simulation")
    assert [port.name() for port in node.input_ports()] == [
        item.port_name for item in spec.inputs
    ]
    assert [port.name() for port in node.output_ports()] == [
        item.port_name for item in spec.outputs
    ]
    assert all(
        port.connected_ports()
        for port in node.input_ports() + node.output_ports()
    )
    assert node.get_property("simulation_interface") == spec.to_json()
    assert len(widget.current_graph.serialize_session()["nodes"]) == (
        1 + len(spec.inputs) + len(spec.outputs)
    )

    model = widget.get_compiled_code()[0]
    assert not model["code"].startswith("# Error")
    namespace = {"__file__": str(REPO_ROOT / "_compiled_model_test.py")}
    exec(model["code"], namespace)

    calls = []

    class FakeCad:
        @staticmethod
        def fea(path, _settings=None, **parameters):
            calls.append((path, parameters))
            return SimpleNamespace(
                max_stress=250.0,
                peak_disp=0.42,
                mass=0.007,
            )

    namespace["cad"] = FakeCad
    values = {item.port_name: item.default for item in spec.inputs}
    result = namespace["system_function_0"](**values)
    assert result == {
        "max_stress": 250.0,
        "peak_disp": 0.42,
        "mass": 0.007,
    }
    assert calls[0][1]["H"] == 10.0

    widget.close()
    widget.deleteLater()
    qapp.processEvents()


def test_simulation_function_round_trips_through_system_file(qapp, tmp_path):
    spec = make_default_spec(inspect_design_studio_study(STUDY_PATH))
    source = ModelingWidget()
    source.create_design_studio_function(
        spec,
        destination_system="Round Trip",
        create_system=True,
        create_io_nodes=True,
    )
    qapp.processEvents()
    path = tmp_path / "systems.json"
    save_graph_to_file(source, str(path))

    restored = ModelingWidget()
    load_graph_from_file(restored, str(path))
    qapp.processEvents()
    simulation_nodes = [
        node
        for node in restored.current_graph.all_nodes()
        if node.type_.startswith("com.pfd.custom_block.simulation")
    ]
    assert len(simulation_nodes) == 1
    node = simulation_nodes[0]
    assert node.get_property("simulation_interface") == spec.to_json()
    assert [port.name() for port in node.input_ports()] == [
        item.port_name for item in spec.inputs
    ]
    assert [port.name() for port in node.output_ports()] == [
        item.port_name for item in spec.outputs
    ]
    assert all(
        port.connected_ports()
        for port in node.input_ports() + node.output_ports()
    )

    source.close()
    source.deleteLater()
    restored.close()
    restored.deleteLater()
    qapp.processEvents()
