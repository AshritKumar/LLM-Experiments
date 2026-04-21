from graphviz import Digraph

class ExpViz:
    # root is a Value object
    def trace(root):
        nodes, edges = set(), set()
        def build_nodes_n_edges(node):
            try:
                if node not in nodes:
                    nodes.add(node)
                    # iterate over children and build recursively
                    for child in node._prev:
                        edges.add((child, node)) # goes from child -> parent
                        build_nodes_n_edges(child)
            except Exception:
                print(f"**Error processing node {type(node)} {node}")
        build_nodes_n_edges(root)
        return nodes, edges

    def viz_expr(root, format='svg', rankdir='LR', view=True):
        nodes, edges = ExpViz.trace(root)
        dot = Digraph(format=format, graph_attr={'rankdir': rankdir})

        # create nodes in graph
        for node in nodes:
            # define nodes - include label if available
            node_label = f"{node.label}|" if node.label else ""
            dot.node(name=str(id(node)), label=f"{{{node_label}data=%.4f}}|{{grad=%.4f}}" % (node.data, node.grad), shape='record')
            # define dynamic nodes to represent operations, leaf nodes won't have op
            if node._op:
                op_name = str(id(node)) + node._op
                dot.node(name=op_name, label=node._op)
                # add an edge b/w op and the node op -> node
                dot.edge(op_name, str(id(node)))

        # connect the edges
        for node1,node2 in edges:
            dot.edge(str(id(node1)), str(id(node2)) + node2._op)
        
        # Optionally display immediately
        if view:
            dot.view()
            
        return dot