# vim: set expandtab shiftwidth=4 softtabstop=4:
"""
ChimeraX DAQ Score Plugin

Provides DAQ score computation and visualization for cryo-EM models.
"""
from chimerax.core.toolshed import BundleAPI


def _release_daq_models(*_args, **_kwargs):
    """Free cached ORT/MLX models before Python interpreter teardown.

    ORT-CUDA InferenceSession holds the CUDA context + workspace; if the
    interpreter exits while the session is still referenced, the CUDA
    driver cleanup ordering can deadlock and the chimerax process hangs
    after the GUI closes. Calling clear_model_cache() here releases the
    sessions while the bundle module is still importable.
    """
    try:
        from .onnx_model import clear_model_cache
        clear_model_cache()
    except Exception:
        pass


class _API(BundleAPI):
    api_version = 1

    @staticmethod
    def initialize(session, bi):
        """Wire app-quit cleanup. Fires once per session startup."""
        try:
            session.triggers.add_handler("app quit", _release_daq_models)
        except Exception:
            # Older ChimeraX may not have the 'app quit' trigger; the
            # bundle remains functional, only the shutdown-hang fix is
            # skipped.
            pass

    @staticmethod
    def finish(session, bi):
        """Called when the bundle is unloaded; release backends now."""
        _release_daq_models()

    #GUI tool
    @staticmethod
    def start_tool(session, bi, ti):
        # ti.name in bundle_info.xml
        if ti.name == "DAQplugin":
            from .tool import DAQTool  # GUI ToolInstance
            return DAQTool(session, ti.name)

        raise ValueError(f"Unknown tool: {ti.name}")
    
    @staticmethod
    def get_class(class_name):
        if class_name == "DAQTool":
            from .tool import DAQTool
            return DAQTool
        raise ValueError(f"Unknown class name '{class_name}'")
    
    @staticmethod
    def register_command(bi, ci, logger):
        print(f"DEBUG: Registering command {ci.name}")
        from . import cmd
        
        # Visualization commands (daqcolor)
        if ci.name == "daqcolor apply":
            func, desc = cmd.daqcolor_apply, cmd.daqcolor_apply_desc
        elif ci.name == "daqcolor monitor":
            func, desc = cmd.daqcolor_monitor, cmd.daqcolor_monitor_desc
        elif ci.name == "daqcolor points":
            func, desc = cmd.daqcolor_points, cmd.daqcolor_points_desc
        elif ci.name == "daqcolor clear":
            func, desc = cmd.daqcolor_clear, cmd.daqcolor_clear_desc
        
        # Computation commands (daqscore)
        elif ci.name == "daqscore compute_grid":
            func, desc = cmd.daqscore_compute_grid, cmd.daqscore_compute_grid_desc
        elif ci.name == "daqscore compute_pdb":
            func, desc = cmd.daqscore_compute_pdb, cmd.daqscore_compute_pdb_desc

        # Arrow window commands (arrowwin)
        elif ci.name == "daq arrowwin":
            func, desc = cmd.daq_arrowwin, cmd.daq_arrowwin_desc
        # Clear DAQ restraints command
        elif ci.name == "daq clearrestraints":
            func, desc = cmd.daq_clearrestraints, cmd.daq_clearrestraints_desc

        else:
            raise ValueError(f"Unknown command: {ci.name}")

        if desc.synopsis is None:
            desc.synopsis = ci.synopsis

        from chimerax.core.commands import register
        register(ci.name, desc, func, logger=logger)


bundle_api = _API()
