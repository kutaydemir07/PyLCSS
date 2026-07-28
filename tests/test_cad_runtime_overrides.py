from pylcss.design_studio.runtime import _apply_property_overrides


class _FakePressureNode:
    __identifier__ = "com.cad.sim.pressure_load"

    def __init__(self):
        self.id = "runtime-generated-id"
        self._pylcss_saved_node_id = "0x00d"
        self._properties = {"pressure": 1.0}

    def name(self):
        return "Pressure (Hole)"

    def has_property(self, name):
        return name in self._properties

    def get_property(self, name):
        return self._properties[name]

    def set_property(self, name, value):
        self._properties[name] = value


class _FakeGraph:
    def __init__(self, node):
        self.node = node

    def all_nodes(self):
        return [self.node]


def test_saved_session_id_alias_applies_runtime_override():
    node = _FakePressureNode()
    applied = _apply_property_overrides(
        _FakeGraph(node), {"0x00d::pressure": 2.5}
    )
    assert applied == 1
    assert node.get_property("pressure") == 2.5
    assert node._dirty is True
