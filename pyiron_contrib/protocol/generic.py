# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from __future__ import print_function
from pyiron.base.job.generic import GenericJob
from pyiron_contrib.protocol.utils import IODictionary, InputDictionary, LoggerMixin, Event, EventHandler, \
    Pointer, CrumbType
from abc import ABC, abstractmethod
from numpy import inf

"""
The objective is to iterate over a directed acyclic graph of simulation instructions.
"""

__author__ = "Liam Huber, Dominik Noeger, Jan Janssen"
__copyright__ = "Copyright 2019, Max-Planck-Institut für Eisenforschung GmbH " \
                "- Computational Materials Design (CM) Department"
__version__ = "0.0"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "development"
__date__ = "Aug 16, 2019"


class Vertex(LoggerMixin, ABC):
    """
    A parent class for objects which are valid vertices of our directed acyclic graph.

    Attributes:
        input (InputDictionary): A pointer-capable dictionary for inputs, including a sub-dictionary for defaults.
            (Default is a clean dictionary.)
        output (IODictionary): A pointer-capable dictionary for outputs. (Default is a clean dictionary.)
        archive (IODictionary): A pointer-capable dictionary for sampling the history of inputs and outputs. (Default
            is a clean dictionary.)
        vertex_state (str): Which edge to follow out of this vertex. (Default is "next".)
        possible_vertex_states (list[str]): Allowable exiting edge names. (Default is ["next"], one exit only!)
        name (str): Vertex name.
        n_history (int): The length of each list stored in the output dictionary.
        on (bool): Whether to execute the vertex when it is the active vertex of the graph, or simply skip over it.
            (default is True -- actually execute!)

    Input attributes:
        default (IODictionary): A dictionary for fall-back values in case a key is requested that isn't in the main
            input dictionary.

    Archive attributes:
        period (int): How frequently to store input and output in the archive. Stores when `clock` % `period` = 0.
        clock (int):
    """

    def __init__(self, name=None):
        self.input = InputDictionary()
        self.output = IODictionary()
        self.archive = IODictionary()
        self.archive.period = inf
        self.archive.clock = 0
        self.archive.output = IODictionary()
        self.archive.input = IODictionary()
        self._vertex_state = "next"
        self.possible_vertex_states = ["next"]
        self._name = None
        self.name = name
        self._n_history = 1
        self.on = True
        self.graph_parent = None

    def get_graph_location(self):
        return self._get_graph_location()[:-1]  # Cut the trailing underscore

    def _get_graph_location(self, loc=""):
        new_loc = self.name + "_" + loc
        if self.graph_parent is None:
            return new_loc
        else:
            return self.graph_parent._get_graph_location(loc=new_loc)

    @property
    def vertex_state(self):
        return self._vertex_state

    @vertex_state.setter
    def vertex_state(self, new_state):
        if new_state not in self.possible_vertex_states:
            raise ValueError("New state not in list of possible states")
        self._vertex_state = new_state

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        self._name = new_name
    # Pyiron was angry when Vertex set `self.name` in `from_hdf`, so we'll weasel around it

    @property
    def n_history(self):
        return self._n_history

    @n_history.setter
    def n_history(self, n_hist):
        self._n_history = n_hist

    @abstractmethod
    def execute(self):
        """What to do when this vertex is the active vertex during graph traversal."""
        pass

    def _update_output(self, output_data):
        if output_data is None:
            return

        for key, value in output_data.items():
            if key not in self.output:
                self.output[key] = [value]
            else:
                history = list(self.output[key])
                # Roll the list if it is necessary
                history.append(value)
                if len(history) > self.n_history:
                    # Remove the head of the queue
                    history.pop(0)
                self.output[key] = history

    def _update_archive(self):
        # Update input
        for key, value in self.input.items():
            if key not in self.archive.input:
                self.archive.input[key] = [value]
            else:
                history = list(self.archive.input[key])
                history.append(value)
                self.archive.input[key] = history
                # self.archive.input[key].append(value)
                # TODO: This will get expensive for large histories, but a direct append doesn't work. Fix it.

        # Update output
        for key, value in self.output.items():
            val = value[-1]
            if key not in self.archive.output:
                self.archive.output[key] = [val]
            else:
                history = list(self.archive.output[key])
                history.append(val)
                self.archive.output[key] = history
                # self.archive.output[key].append(val)

    def update_and_archive(self, output_data):
        self._update_output(output_data)
        if self.archive.clock % self.archive.period == 0:
            self._update_archive()

    def finish(self):
        self._update_archive()

    def parallel_setup(self):
        """How to prepare to execute in parallel when there's a list of these vertices together."""
        pass

    def to_hdf(self, hdf, group_name=None):
        """
        Store the Vertex in an HDF5 file.

        Args:
            hdf (ProjectHDFio): HDF5 group object
            group_name (str): HDF5 subgroup name
        """
        if group_name is None:
            hdf["TYPE"] = str(type(self))
            hdf["possiblevertexstates"] = self.possible_vertex_states
            hdf["vertexstate"] = self.vertex_state
            hdf["name"] = self.name
            hdf["nhistory"] = self._n_history
            self.input.to_hdf(hdf=hdf, group_name="input")
            self.output.to_hdf(hdf=hdf, group_name="output")
            self.archive.to_hdf(hdf=hdf, group_name="archive")
        else:
            with hdf.open(group_name) as hdf5_server:
                hdf5_server["TYPE"] = str(type(self))
                hdf5_server["possiblevertexstates"] = self.possible_vertex_states
                hdf5_server["vertexstate"] = self.vertex_state
                hdf5_server["name"] = self.name
                hdf5_server["nhistory"] = self._n_history
                self.input.to_hdf(hdf=hdf5_server, group_name="input")
                self.output.to_hdf(hdf=hdf5_server, group_name="output")
                self.archive.to_hdf(hdf=hdf5_server, group_name="archive")

    def from_hdf(self, hdf, group_name=None):
        """
        Load the Vertex from an HDF5 file.

        Args:
            hdf (ProjectHDFio): HDF5 group object
            group_name (str): HDF5 subgroup name
        """
        if group_name is None:
            self.possible_vertex_states = hdf["possiblevertexstates"]
            self._vertex_state = hdf["vertexstate"]
            self._name = hdf["name"]
            self._n_history = hdf["nhistory"]
            self.input.from_hdf(hdf=hdf, group_name="input")
            self.output.from_hdf(hdf=hdf, group_name="output")
            self.archive.from_hdf(hdf=hdf, group_name="archive")
        else:
            with hdf.open(group_name) as hdf5_server:
                self.possible_vertex_states = hdf5_server["possiblevertexstates"]
                self._vertex_state = hdf5_server["vertexstate"]
                self._name = hdf5_server["name"]
                self._n_history = hdf5_server["nhistory"]
                self.input.from_hdf(hdf=hdf5_server, group_name="input")
                self.output.from_hdf(hdf=hdf5_server, group_name="output")
                self.archive.from_hdf(hdf=hdf5_server, group_name="archive")


class PrimitiveVertex(Vertex):
    """
    Vertices which do one thing.
    """

    def execute(self):
        """Just parse the input and do your physics, then store the output."""
        output_data = self.command(**self.input.resolve())
        self.update_and_archive(output_data)

    @abstractmethod
    def command(self, *args, **kwargs):
        """The command method controls the physics"""
        pass

    def execute_parallel(self, queue, n, input_dict):
        """How to execute in parallel when there's a list of these vertices together."""
        output_data = self.command(**input_dict)
        queue.put((n, output_data))
        # Note: The output needs to be explicitly collected and archived later if this is used in place of `execute`


class Protocol(Vertex, GenericJob):
    """
    Can either be the parent graph to execute (when given a project and job name at instantiation, e.g. when created as
    a pyiron job), or a vertex which contains its own graph and has its own sub-vertices.

    Input:
        graph (Graph): The graph of vertices to traverse.
        protocol_finished (Event):
        protocol_started (Event):
        vertex_processing (Event):
        vertex_processed (Event):
        finished (bool):
    """
    def __init__(self, project=None, name=None, job_name=None):
        if name is not None and job_name is not None:
            raise ValueError("Only one of `name` and `job_name` may be set, but `name`={} and `job_name`={}".format(
                name, job_name
            ))
        name = job_name or name
        # When submitting, instantiating a GenericJob requires a `job_name` variable
        self._is_master = False
        if project is not None:
            self._is_master = True
            GenericJob.__init__(self, project, name)
        Vertex.__init__(self, name=name)

        self.graph = Graph()
        self.graph.owner = self

        # Initialize event system
        self.protocol_finished = Event()
        self.protocol_started = Event()
        self.vertex_processing = Event()
        self.vertex_processed = Event()

        # Set up the graph
        self.define_vertices()
        self.define_execution_flow()
        self.define_information_flow()

        # On initialization, set the active verex to starting vertex
        self.graph.active_vertex = self.graph.starting_vertex

        self.finished = False

    @property
    def n_history(self):
        return self._n_history

    @n_history.setter
    def n_history(self, n_hist):
        self._n_history = n_hist
        for _, vertex in self.graph.vertices.items():
            vertex.n_history = n_hist

    @abstractmethod
    def define_vertices(self):
        pass

    @abstractmethod
    def define_execution_flow(self):
        pass

    @abstractmethod
    def define_information_flow(self):
        pass

    @abstractmethod
    def get_output(self):
        pass

    def execute(self):
        """Traverse graph until the active vertex is None."""
        # Subscribe graph vertices to the protocol_finished Event
        for vertex_name, vertex in self.graph.vertices.items():
            handler_name = '{}_close_handler'.format(vertex_name)
            if not self.protocol_finished.has_handler(handler_name):
                self.protocol_finished += EventHandler(handler_name, vertex.finish)

        # Run the graph
        if self.graph.active_vertex is None:
            self.graph.active_vertex = self.graph.starting_vertex
        self.protocol_started.fire()
        while self.graph.active_vertex is not None:
            vertex_on = self.graph.active_vertex.on
            if isinstance(vertex_on, Pointer):
                vertex_on = ~vertex_on
            if not vertex_on:
                self.logger.info('Skipping vertex "{}":{}'.format(self.graph.active_vertex.name,
                                                                  type(self.graph.active_vertex).__name__))
                continue
            self.logger.info('Executing vertex "{}":{}'.format(self.graph.active_vertex.name,
                                                               type(self.graph.active_vertex).__name__))
            self.vertex_processing.fire(self.graph.active_vertex)
            self.graph.active_vertex.execute()
            self.vertex_processed.fire(self.graph.active_vertex)
            self.graph.step()
        self.graph.active_vertex = self.graph.restarting_vertex
        self.update_and_archive(self.get_output())

        if self._is_master:
            self.protocol_finished.fire()

    def execute_parallel(self, queue, n, input):
        """How to execute in parallel when there's a list of these vertices together."""
        self.execute()
        queue.put((n, self.get_output()))

    def set_graph_archive_period(self, period):
        for _, vertex in self.graph.vertices.items():
            vertex.archive.period = period

    def set_graph_archive_clock(self, clock):
        for _, vertex in self.graph.vertices.items():
            vertex.archive.clock = clock

    def run_static(self):
        """If this CompoundVertex is the highest level, it can be run as a regular pyiron job."""
        self.status.running = True
        self.execute()
        self.status.collect = True  # Assume modal for now
        self.run()  # This is an artifact of inheriting from GenericJob, to get all that run functionality

    def collect_output(self):
        # Dear Reader: This feels like a hack, but it works. Sincerely, -Liam
        self.to_hdf()

    def write_input(self):
        # Dear Reader: I looked at base/master/list and /parallel where this appears, but it's still not clear to me
        # what I should be using this for. But, I get a NotImplementedError if I leave it out, so here it is. -Liam
        pass

    def to_hdf(self, hdf=None, group_name=None):
        """
        Store the Protocol in an HDF5 file.

        Args:
            hdf (ProjectHDFio): HDF5 group object - optional
            group_name (str): HDF5 subgroup name - optional
        """
        if hdf is None:
            hdf = self.project_hdf5
        if self._is_master:
            GenericJob.to_hdf(self, hdf=hdf, group_name=group_name)
        Vertex.to_hdf(self, hdf=hdf, group_name=group_name)
        self.graph.to_hdf(hdf=hdf, group_name="graph")
        try:
            hdf[group_name]["ismaster"] = self._is_master
        except AttributeError:
            hdf["ismaster"] = self._is_master

    def from_hdf(self, hdf=None, group_name=None):
        """
        Load the Protocol from an HDF5 file.

        Args:
            hdf (ProjectHDFio): HDF5 group object - optional
            group_name (str): HDF5 subgroup name - optional
        """
        if hdf is None:
            hdf = self.project_hdf5
        try:
            self._is_master = hdf[group_name]["ismaster"]
        except AttributeError:
            self._is_master = hdf["ismaster"]
        if self._is_master:
            GenericJob.from_hdf(self, hdf=hdf, group_name=group_name)
        Vertex.from_hdf(self, hdf=hdf, group_name=group_name)
        self.graph.from_hdf(hdf=hdf, group_name="graph")
        self.define_information_flow()  # Rewire pointers

    def visualize(self, execution=True, dataflow=True):
        return self.graph.visualize(self.fullname(), execution=execution, dataflow=dataflow)


class Graph(dict, LoggerMixin):
    """
    A directed graph of vertices and edges, and a method for iterating through the graph.

    Vertices and edges are the graph are explicitly stored as child classes inheriting from `dict` so that all
    'graphiness' is fully decoupled from the objects sitting at the vertices.

    Attributes:
        vertices (Vertices): Vertices of the graph.
        edges (Edges): Directed edges between the vertices.
        starting_vertex (Vertex): The element of `vertices` for the graph to begin on.
        active_vertex (Vertex): The element of `vertices` for the vertex the graph iteration is on.
        restarting_vertex (Vertex): The element of `vertices` for the graph to restart on if the graph has been loaded.
    """

    def __init__(self, **kwargs):
        super(Graph, self).__init__(**kwargs)
        self.vertices = Vertices()
        self.edges = Edges()
        self.starting_vertex = None
        self.active_vertex = None
        self.restarting_vertex = None
        self.owner = None

    def __setattr__(self, key, val):
        if key == "vertices":
            if not isinstance(val, Vertices):
                raise ValueError("'vertices' is a protected attribute for graphs.")
            self[key] = val
        elif key == "edges":
            if not isinstance(val, Edges):
                raise ValueError("'edges' is a protected attribute for graphs.")
            self[key] = val
        elif key in ["active_vertex", "starting_vertex", "restarting_vertex"]:
            if val is None or isinstance(val, Vertex):
                self[key] = val
            else:
                raise ValueError("The active, starting, and restarting vertices must inherit `Vertex` or be `None`.")
        elif key == "owner":
            if not (isinstance(val, Protocol) or val is None):
                raise ValueError("Only protocols can hold graphs, but the assigned owner has type", type(val))
            else:
                self[key] = val
        elif isinstance(val, Vertex):
            val.name = key
            val.graph_parent = self.owner
            self.vertices[key] = val
            self.edges.initialize(val)
        else:
            raise TypeError("Graph vertices must inherit from `Vertex`")

    def __getattr__(self, name):
        try:
            return self["vertices"][name]
        except KeyError:
            try:
                return self[name]
            except KeyError:
                return object.__getattribute__(self, name)

    def visualize(self, protocol_name, execution=True, dataflow=True):
        """
        Plot a visual representation of the graph.

        Args:
            protocol_name:
            execution (bool): Show the lines dictating the flow of graph traversal.
            dataflow (bool): Show the lines dictating where vertex input comes from.

        Returns:
            (graphviz.Digraph) The image representation of the protocols workflow
        """
        try:
            from graphviz import Digraph
        except ImportError as import_error:
            self.logger.exception('Failed to import "graphviz" package', exc_info=import_error)
            return

        # Create graph object
        workflow = Digraph(comment=protocol_name)

        # Define styles for the individual classes
        class_style_mapping = {
            Protocol: {'shape': 'box'},
            # CommandBool: {'shape': 'diamond'},
            PrimitiveVertex: {'shape': 'circle'}
        }

        def resolve_type(type_):
            if type_ in class_style_mapping.keys():
                return type_
            else:
                parents = [key for key in class_style_mapping.keys() if issubclass(type_, key)]
                if len(parents) == 0:
                    raise TypeError('I do not know how to visualize "{}"'.format(type_.__name__))
                elif len(parents) > 1:
                    self.logger.warn('More than one parent class found for type "{}"'.format(type_.__name__))
                return parents[0]

        for vertex_name, vertex in self.vertices.items():
            vertex_type = type(vertex)
            vertex_type_style = class_style_mapping[resolve_type(vertex_type)]

            node_label = '''<<B>{vertex_type}</B><BR/>{vertex_name}>'''
            node_label = node_label.format(vertex_type=vertex_type.__name__,
                                           vertex_name=vertex_name)
            if self.active_vertex == vertex:
                # Highlight active vertex
                highlight = {
                    'style': 'filled',
                    'color': 'green',
                }
            else:
                highlight = {}

            # Highlight the active vertex in green color
            highlight.update(vertex_type_style)
            workflow.node(vertex_name, label=node_label, **highlight)
        # Add end node
        workflow.node('end', 'END', **{'shape': 'doublecircle', 'style': 'filled', 'color': 'red'})
        protocol_input_node = None

        if execution:
            for vertex_start, edges in self.edges.items():
                for vertex_state, vertex_end in edges.items():
                    if vertex_end is None:
                        vertex_end = 'end'
                    workflow.edge(vertex_start, vertex_end, label=vertex_state)
        if dataflow:
            dataflow_edge_style = {
                'style': 'dotted',
                'color': 'blue',
                'labelfontcolor': 'blue',
                'labelangle': '90'
            }
            for vertex_name, vertex in self.vertices.items():
                items = super(InputDictionary, vertex.input).items()
                for key, value in items:
                    if isinstance(value, Pointer):
                        vertex_end = self._edge_from_pointer(key, value)
                        if vertex_end is not None:
                            if isinstance(vertex_end, Vertex):
                                workflow.edge(vertex_end.name, vertex_name, label=key, **dataflow_edge_style)
                            elif isinstance(vertex_end, (IODictionary, Vertex)):
                                self.logger.warning('vertex_end is IODIctionary() I have to decide what to do')
                                if protocol_input_node is None:
                                    # Initialize a node for protocol level input
                                    protocol_input_node = workflow.node(
                                        'protocol_input', '{}.input'.format(protocol_name),
                                        style='filled',
                                        shape='folder'
                                    )
                                workflow.edge('protocol_input', vertex_name, label=key, **dataflow_edge_style)
                            else:
                                pass

        return workflow

    def _edge_from_pointer(self, key, p):
        assert isinstance(p, Pointer)
        path = p.path.copy()
        root = path.pop(0)

        result = root.object
        while len(path) > 0:
            crumb = path.pop(0)
            crumb_type = crumb.crumb_type
            crumb_name = crumb.name

            if isinstance(result, (Vertex, IODictionary)):
                return result

            # If the result is a pointer itself we have to resolve it first
            if isinstance(result, Pointer):
                self.logger.info('Resolved pointer in a pointer path')
                result = ~result
            if isinstance(crumb_name, Pointer):
                self.logger.info('Resolved pointer in a pointer path')
                crumb_name = ~crumb_name
            # Resolve it with the correct method - dig deeper
            if crumb_type == CrumbType.Attribute:
                try:
                    result = getattr(result, crumb_name)
                except AttributeError as e:
                    raise e
            elif crumb_type == CrumbType.Item:
                try:
                    result = result.__getitem__(crumb_name)
                except (TypeError, KeyError) as e:
                    raise e

        # If we reached this point we have no Command at all, give a warning
        self.logger.warning('I could not find a graph in the pointer {} for key "{}"'.format(p.path, key))
        return None

    def step(self):
        """
        Follows the edge out of the active vertex to get the name of the next vertex and set it as the active vertex.
        If the active vertex has multiple possible states, the outbound edge for the current state will be chosen.

        Returns:
            (str) The name of the next vertex.
        """
        vertex = self.active_vertex
        if vertex is not None:
            state = vertex.vertex_state
            next_vertex_name = self.edges[vertex.name][state]

            if next_vertex_name is None:
                self.active_vertex = None
            else:
                self.active_vertex = self.vertices[next_vertex_name]

    def make_edge(self, start, end, state="next"):
        """
        Makes a directed edge connecting two vertices.

        Args:
            start (Vertex): The vertex for the edge to start at.
            end (Vertex): The vertex for the edge to end at.
            state (str): The state for the vertex to be in when it points to this particular end. (Default, "next", is
                the parent-level state for vertices without multiple outbound edges.)
        """
        assert(start.name in self.vertices.keys())

        if state != "next":
            assert(state in start.possible_vertex_states)

        if end is not None:
            assert(end.name in self.vertices.keys())
            self.edges[start.name][state] = end.name
        else:
            self.edges[start.name][state] = None

    def make_pipeline(self, *args):
        """
        Adds an edge between every argument, in the order they're given. The edge is added for the vertex state "next",
        so this is only appropriate for vertices which don't have a non-trivial `vertex_state`.

        Args:
            *args (Vertex/str): Vertices to connect in a row, or the state connecting two vertices.
        """
        for n, vertex in enumerate(args[:-1]):
            if isinstance(vertex, str):
                continue
            next_vertex = args[n + 1]
            if isinstance(next_vertex, str):
                state = next_vertex
                next_vertex = args[n + 2]
            else:
                state = "next"
            self.make_edge(vertex, next_vertex, state=state)

    def to_hdf(self, hdf, group_name="graph"):
        with hdf.open(group_name) as hdf5_server:
            hdf5_server["TYPE"] = str(type(self))
            hdf5_server["startingvertexname"] = self.starting_vertex.name
            if self.active_vertex is not None:
                hdf5_server["activevertexname"] = self.active_vertex.name
            else:
                hdf5_server["activevertexname"] = None
            if self.restarting_vertex is None:
                hdf5_server["restartingvertexname"] = self.starting_vertex.name
            else:
                hdf5_server["restartingvertexname"] = self.restarting_vertex.name
            self.vertices.to_hdf(hdf=hdf5_server, group_name="vertices")
            self.edges.to_hdf(hdf=hdf5_server, group_name="edges")

    def from_hdf(self, hdf, group_name="graph"):
        with hdf.open(group_name) as hdf5_server:
            active_vertex_name = hdf5_server["activevertexname"]
            self.vertices.from_hdf(hdf=hdf5_server, group_name="vertices")
            self.edges.from_hdf(hdf=hdf5_server, group_name="edges")
            self.starting_vertex = self.vertices[hdf5_server["startingvertexname"]]
            self.restarting_vertex = self.vertices[hdf5_server["restartingvertexname"]]
        if active_vertex_name is not None:
            self.active_vertex = self.vertices[active_vertex_name]
        else:
            self.active_vertex = self.restarting_vertex


class Vertices(dict):
    """
    A dictionary of vertices whose keys are the vertex name.
    """

    def __init__(self, **kwargs):
        super(Vertices, self).__init__(**kwargs)

    def __setattr__(self, key, val):
        if not isinstance(val, Vertex):
            raise ValueError
        self[key] = val

    def __getattr__(self, item):
        return self[item]

    def to_hdf(self, hdf, group_name="vertices"):
        with hdf.open(group_name) as hdf5_server:
            hdf5_server["TYPE"] = str(type(self))
            for name, vertex in self.items():
                if isinstance(vertex, Vertex):
                    vertex.to_hdf(hdf=hdf5_server, group_name=name)
                else:
                    raise TypeError("Cannot save non-Vertex-like vertices")

    def from_hdf(self, hdf, group_name="vertices"):
        with hdf.open(group_name) as hdf5_server:
            for name, vertex in self.items():
                if isinstance(vertex, Vertex):
                    vertex.from_hdf(hdf=hdf5_server, group_name=name)
                else:
                    raise TypeError("Cannot load non-Vertex-like vertices")


class Edges(dict):
    """
    A collection of dictionaries connecting each state of a given vertex to another vertex.
    """

    def __init__(self, **kwargs):
        super(Edges, self).__init__(**kwargs)

    def __setattr__(self, key, val):
        self[key] = val

    def __getattr__(self, item):
        return self[item]

    def initialize(self, vertex):
        """
        Set an outbound edge to `None` for each allowable vertex state.

        Args:
            vertex (Vertex): The vertex to assign an edge to.
        """
        name = vertex.name
        if isinstance(vertex, Vertex):
            self[name] = {}
            for state in vertex.possible_vertex_states:
                self[name][state] = None
        else:
            raise TypeError("Vertices must inherit from `Vertex` .")

    def to_hdf(self, hdf, group_name="edges"):
        with hdf.open(group_name) as hdf5_server:
            hdf5_server["TYPE"] = str(type(self))
            for name, edge in self.items():
                hdf5_server[name] = edge

    def from_hdf(self, hdf, group_name):
        return
