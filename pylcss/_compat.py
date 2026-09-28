"""Resolve legacy module names to the same canonical workspace modules."""
import importlib
import importlib.abc
import importlib.util
import sys

ALIASES = {'pylcss.optimization': 'pylcss.mdao.optimization', 'pylcss.sensitivity': 'pylcss.mdao.sensitivity', 'pylcss.solution_space': 'pylcss.mdao.solution_space', 'pylcss.surrogate_modeling': 'pylcss.mdao.surrogate_modeling', 'pylcss.system_modeling': 'pylcss.mdao.system_modeling', 'pylcss.design_studio': 'pylcss.cae.design_studio', 'pylcss.solver_backends': 'pylcss.cae.solver_backends', 'pylcss.user_interface.optimization': 'pylcss.mdao.user_interface.optimization', 'pylcss.user_interface.sensitivity': 'pylcss.mdao.user_interface.sensitivity', 'pylcss.user_interface.solution_space': 'pylcss.mdao.user_interface.solution_space', 'pylcss.user_interface.surrogate': 'pylcss.mdao.user_interface.surrogate', 'pylcss.user_interface.system_modeling': 'pylcss.mdao.user_interface.system_modeling', 'pylcss.user_interface.cad': 'pylcss.cae.user_interface.cad', 'pylcss.user_interface.main_application_window': 'pylcss.mdao.user_interface.main_application_window'}

class _AliasLoader(importlib.abc.Loader):
    def __init__(self, target):
        self.target = target
    def create_module(self, spec):
        module = importlib.import_module(self.target)
        self.canonical_spec = module.__spec__
        return module
    def exec_module(self, module):
        module.__spec__ = self.canonical_spec

class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        for old, new in ALIASES.items():
            if fullname == old or fullname.startswith(old + "."):
                canonical = new + fullname[len(old):]
                spec = importlib.util.find_spec(canonical)
                if spec is not None:
                    return importlib.util.spec_from_loader(
                        fullname, _AliasLoader(canonical),
                        is_package=spec.submodule_search_locations is not None,
                    )
        return None

def install():
    if not any(isinstance(finder, _AliasFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _AliasFinder())
