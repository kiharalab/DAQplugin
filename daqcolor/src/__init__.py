# vim: set expandtab shiftwidth=4 softtabstop=4:
"""
ChimeraX DAQ Score Plugin

Provides DAQ score computation and visualization for cryo-EM models.
"""
from chimerax.core.toolshed import BundleAPI


class _API(BundleAPI):
    api_version = 1

    @staticmethod
    def register_command(bi, ci, logger):
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

        else:
            raise ValueError(f"Unknown command: {ci.name}")

        if desc.synopsis is None:
            desc.synopsis = ci.synopsis

        from chimerax.core.commands import register
        register(ci.name, desc, func, logger=logger)


bundle_api = _API()
