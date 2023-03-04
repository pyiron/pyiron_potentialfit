from .container import (
            StructureInput,
            AbstractOutput
)
from . import AbstractNode, ListNode
from . import ReturnStatus

from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

MurnaghanInputBase = StructureInput.from_attributes(
        "MurnaghanInputBase",
        "strains",
        "node"
)

class MurnaghanInput(MurnaghanInputBase):
    def check_ready(self):
        structure_ready = self.structure is not None
        strain_ready = len(self.strains) > 0
        node = self.node
        node.input.structure = self.structure
        return structure_ready and strain_ready and node.input.check_ready()

    def set_strain_range(self, range, steps):
        self.strains = (1 + np.linspace(-range, range, steps))**(1/3)

MurnaghanOutputBase = AbstractOutput.from_attributes(
        "MurnaghanOutputBase",
        volumes=list,
        energies=list
)

class MurnaghanOutput(MurnaghanOutputBase):
    def plot(self):
        plt.plot(self.volumes, self.energies)

class MurnaghanNode(ListNode):

    def _get_input(self):
        return MurnaghanInput()

    def _get_output(self):
        return MurnaghanOutput()

    def _create_nodes(self):
        cell = self.input.structure.get_cell()
        nodes = []
        for s in self.input.strains:
            n = deepcopy(self.input.node)
            n.input.structure = self.input.structure.copy()
            n.input.structure.set_cell(cell * s, scale_atoms=True)
            nodes.append(n)
        return nodes

    def _extract_output(self, i, node, ret):
        if ret.is_done():
            self.output.energies.append(node.output.energy_pot)
            self.output.volumes.append(node.input.structure.get_volume())
