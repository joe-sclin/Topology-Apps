import sys
import pandas as pd
import networkx as nx
import pulp
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QAction
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Visualization of directed graph")
        self.setGeometry(100, 100, 1280, 720)
        self.G = nx.DiGraph()

        file_menu = self.menuBar().addMenu("&File")
        open_action = QAction("Open Graph", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_file_dialog)
        file_menu.addAction(open_action)

        closeGraphAction = QAction('Close Graph', self)
        closeGraphAction.triggered.connect(self.closeGraph)
        file_menu.addAction(closeGraphAction)

        # Figure canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.addToolBar(self.toolbar)
        self.setCentralWidget(self.canvas)

        edit_menu = self.menuBar().addMenu("&Edit")
        remove_self_loop_action = QAction("Remove self loop", self)
        remove_self_loop_action.triggered.connect(lambda: self.remove_self(self.G))
        edit_menu.addAction(remove_self_loop_action)
        reversed_graph = QAction("Reverse graph", self)
        reversed_graph.triggered.connect(lambda: self.reversed_direction(self.G))
        edit_menu.addAction(reversed_graph)

        compute_menu = self.menuBar().addMenu("&Compute")
        compute_mds = QAction("MDS", self)
        compute_mds.triggered.connect(lambda: self.run_topological_feature(self.G, self.mds))
        compute_menu.addAction(compute_mds)
        compute_mcds = QAction("MCDS", self)
        compute_mcds.triggered.connect(lambda: self.run_topological_feature(self.G, self.mcds))
        compute_menu.addAction(compute_mcds)
        compute_indegree = QAction("In-degree Hub", self)
        compute_indegree.triggered.connect(lambda: self.run_topological_feature(self.G, self.in_degree_hubs))
        compute_menu.addAction(compute_indegree)
        compute_outdegree = QAction("Out-degree Hub", self)
        compute_outdegree.triggered.connect(lambda: self.run_topological_feature(self.G, self.out_degree_hubs))
        compute_menu.addAction(compute_outdegree)
        compute_lcc = QAction("LCC", self)
        compute_lcc.triggered.connect(lambda: self.run_topological_feature(self.G, self.lcc))
        compute_menu.addAction(compute_lcc)

    def open_file_dialog(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Edgelist File", "", "CSV Files (*.csv)")
        if filename:
            self.load_graph(filename)

    def load_graph(self, filename):
        df = pd.read_csv(filename, sep='\t')
        self.G = nx.from_pandas_edgelist(df, 'regulator', 'target', create_using=nx.DiGraph)
        self.draw_graph(self.G, [])

    def remove_self(self, G):
        for self_edge in nx.selfloop_edges(G):    # nx function treated self-loop as cycle
            G.remove_edge(self_edge[0], self_edge[1])
        self.closeGraph()
        self.draw_graph(G, [])

    def reversed_direction(self, G):
        temp = G.reverse()
        self.G = temp
        self.closeGraph()
        self.draw_graph(self.G, [])

    def closeGraph(self):
        # Code to close the current graph and clear the canvas goes here
        self.figure.clear()
        self.canvas.draw()

    def draw_graph(self, G, labeled_nodes):
        pos = nx.circular_layout(G)
        node_colors = ['red' if node in labeled_nodes else 'lightblue' for node in G.nodes()]
        node_labels = {node: node if node in labeled_nodes else '' for node in G.nodes()}
        nx.draw(G, pos, with_labels=True, arrows=True, labels=node_labels, font_color='black',
                verticalalignment='bottom', node_color=node_colors, edge_color='lightgray', node_size=8,
                ax=self.figure.add_subplot(111))
        # Set the initial zoom level based on the size of the graph
        self.toolbar.update()
        x_vals, y_vals = zip(*pos.values())
        self.toolbar.zoom(1.0 / max(abs(max(x_vals)), abs(min(x_vals))), 1.0 / max(abs(max(y_vals)), abs(min(y_vals))))
        self.canvas.draw()

    def run_topological_feature(self, G, function):
        if len(G.nodes()) > 0:
            topo_nodes = function(G)
            self.closeGraph()
            self.draw_graph(G, topo_nodes)

    def mds(self, g: nx.DiGraph) -> list:
        """
        Applying ILP algorithm to calculate MDS for a directed / undirected graph
        :param g: networkx Graph / DiGraph object
        :return: List of name of nodes in MDS
        """
        g_reversed = g.reverse(copy=True)  # LpProblem treated all edges in reversed order
        # LpProblem to create LP problem object "prob" (Sense = LpMinimize / LpMaximize)
        prob = pulp.LpProblem("MDS", pulp.LpMinimize)  # Minimize number of nodes selected
        # LpVariable to create LP variable: dict node
        # Key: name of node, Value: LP variable (node_XXX) where XXX = name of node
        # Category of variable: Integer, Binary or Continuous
        node = pulp.LpVariable.dicts("node", g_reversed.nodes(), cat=pulp.LpBinary)
        prob.setObjective(pulp.lpSum(node))  # All nodes are selected (varValue = 1)
        for v in g_reversed.nodes():  # ILP algorithm for MDS
            prob += node[v] + pulp.lpSum([node[u] for u in g_reversed.neighbors(v)]) >= 1
        prob.solve()
        # Inverse mapping to extract correct name of nodes ( '-' in nodes were converted to '_')
        inverse_node = {v.name: k for k, v in node.items()}
        return [inverse_node[v.name] for v in prob.variables() if v.varValue == 1]

    def mcds(self, graph: nx.DiGraph):
        """
        Heuristic approach to determine MCDS for the largest strongly connected component (LSCC) of a directed graph
        :param graph: networkx directed graph
        :returns: lscc_mcds: list of MCDS nodes for LSCC
        """

        def phase1(lcc: nx.DiGraph):
            white = set(lcc.nodes())
            gray = set()
            black = set()
            while len(white) > 0:  # Phase 1
                max_out = max(lcc.out_degree(white), key=lambda x: x[1])[1]  # Max out degree
                max_out_nodes = [x[0] for x in lcc.out_degree(white) if x[1] == max_out]  # Nodes with max out degree
                if len(max_out_nodes) > 1:
                    # Get node with max in degree if multiple nodes have the same max out degree
                    max_in = max(lcc.in_degree(max_out_nodes), key=lambda x: x[1])[1]  # Max out degree
                    dom_nodes = [x[0] for x in lcc.in_degree(max_out_nodes) if
                                 x[1] == max_in]  # Nodes with max out degree
                else:
                    dom_nodes = list(max_out_nodes)
                for node in dom_nodes:
                    black.add(node)
                    # Add all child neighbors of new black node into gray (if they are not in set black)
                    gray.update(set(n for n in lcc.neighbors(node)).difference(black))
                white = white.difference(black.union(gray))  # All remaining nodes are for next round selection
            return black, gray

        def phase2(lcc: nx.DiGraph, black: set, gray: set) -> list:
            dark_gray = []
            black_dark_gray_lcc = lcc.subgraph(black.union(set(dark_gray))).copy()
            while True:
                connected_component = [c for c in nx.weakly_connected_components(black_dark_gray_lcc)]
                gray_arcs_to_black = {}
                gray_arcs_from_black = {}
                for node in gray:
                    # Edges for each gray node to black nodes
                    gray_arcs_to_black[node] = [n for n in lcc.successors(node) if n in black]
                    # Possible dark gray node (max arcs to black nodes)
                max_d_g = len(sorted(gray_arcs_to_black.items(), key=lambda item: len(item[1]), reverse=True)[0][1])
                if max_d_g == 0:  # No remaining gray nodes have edges to black nodes
                    break
                # Possible dark gray nodes
                max_d_g_nodes = [x[0] for x in gray_arcs_to_black.items() if len(x[1]) == max_d_g]
                # Edges exist from black nodes to possible dark gray nodes
                for dg_node in max_d_g_nodes:
                    gray_arcs_from_black[dg_node] = [n for n in lcc.predecessors(dg_node) if n in black]
                # Final filter: max arcs to black nodes + edges exist from black nodes
                filtered_dg_node = [node for node in max_d_g_nodes if
                                    len(gray_arcs_to_black[node]) > 0 and len(gray_arcs_from_black[node]) > 0]
                for dg_node in filtered_dg_node:
                    to_black_cc_list = []  # Index for connected component (possible dark gray to black)
                    for to_black in gray_arcs_to_black[dg_node]:
                        for i, s in enumerate(connected_component):
                            if to_black in s:
                                to_black_cc_list.append(i)
                    from_black_cc_list = []  # Index for connected component (possible dark gray from black)
                    for from_black in gray_arcs_from_black[dg_node]:
                        for i, s in enumerate(connected_component):
                            if from_black in s:
                                from_black_cc_list.append(i)
                    if set(from_black_cc_list) != set(
                            to_black_cc_list):  # If any of index is difference (Connected to a new cc)
                        dark_gray.append(dg_node)
                        gray.remove(dg_node)
                        black_dark_gray_lcc = lcc.subgraph(black.union(set(dark_gray))).copy()
                    else:
                        gray.remove(dg_node)  # Not useful gray node (Only having edges within the same cc)
                if nx.is_weakly_connected(black_dark_gray_lcc):
                    break
            return list(black_dark_gray_lcc.nodes())

        def phase3(cds: list, lcc: nx.DiGraph) -> list:
            node_list = list(cds)  # List of nodes needed to test
            mcds = []
            while len(node_list) > 0:
                min_out = min(lcc.out_degree(node_list), key=lambda x: x[1])[1]  # Min out degree
                min_out_nodes = [x[0] for x in lcc.out_degree(node_list) if
                                 x[1] == min_out]  # Nodes with min out degree
                if len(min_out_nodes) > 1:
                    # Get node with max in degree if multiple nodes have the same min out degree
                    max_in = max(lcc.in_degree(min_out_nodes), key=lambda x: x[1])[1]  # Max in degree
                    max_in_nodes = [x[0] for x in lcc.in_degree(min_out_nodes) if
                                    x[1] == max_in]  # Nodes with min out degree
                    testing_nodes = max_in_nodes
                else:
                    testing_nodes = min_out_nodes
                for node in testing_nodes:
                    temp = node_list.copy()
                    temp.remove(node)
                    current_node = set(temp).union(set(mcds))
                    if not nx.is_weakly_connected(lcc.subgraph(current_node)) or not nx.is_dominating_set(lcc,
                                                                                                          current_node):
                        mcds.append(node)
                    node_list.remove(node)
            return mcds

        scc = nx.strongly_connected_components(graph)
        lscc = max(scc, key=len)
        lcc = graph.subgraph(lscc)
        black, gray = phase1(lcc)
        ds = lcc.subgraph(black)  # For phase 2 checking
        if not nx.is_weakly_connected(ds):  # Enter phase 2
            cds = phase2(lcc, black, gray)  # Connected dominating set (Not minimum yet)
        else:
            cds = ds.nodes()
        if len(cds) > 1:
            min_cds = phase3(cds, lcc)
        else:  # If cds only contains 1 node
            min_cds = cds
        return list(min_cds)

    def out_degree_hubs(self, graph: nx.DiGraph) -> list:
        """
        Calculate outgoing edges degree hubs score for a directed graph
        :param graph:networkx DiGraph object
        :return: Outgoing degree hubs score sorted list of top 10% nodes
        """
        res = sorted(graph.out_degree, key=lambda x: x[1], reverse=True)[:round(len(graph.nodes()) * 0.1)]
        return list(list(zip(*res))[0])

    def in_degree_hubs(self, graph: nx.DiGraph) -> list:
        """
        Calculate incoming edges degree hubs score for a directed graph
        :param graph:networkx DiGraph object
        :return: Outgoing degree hubs score sorted list of top 10% nodes
        """
        res = sorted(graph.in_degree, key=lambda x: x[1], reverse=True)[:round(len(graph.nodes()) * 0.1)]
        return list(list(zip(*res))[0])

    def lcc(self, graph: nx.DiGraph) -> list:
        """
        Find nodes in the largest connected component (in DFS tree) in a directed graph
        :param graph: networkx directed graph object
        :return: networkx directed graph object
        """
        bidirectional_edges = list(set([u and v for u, v in list(graph.edges()) if graph.has_edge(v, u) and graph.has_edge(u, v)]))
        candidate = [x[0] for x in graph.in_degree() if x[1] == 0] + bidirectional_edges
        dfs_tree = nx.dfs_tree(nx.DiGraph(), source=None)
        for x in candidate:
            tree = nx.dfs_tree(graph, source=x)
            if len(tree.nodes()) > len(dfs_tree.nodes()):
                dfs_tree = tree
        return graph.subgraph(dfs_tree.nodes())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
