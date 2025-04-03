import networkx as nx
import matplotlib.pyplot as plt

from .mcts_common import DecisionNode, ChanceNode, chance_node_value
from ...evaluation.utils import hierarchy_pos


def pre_order_traverse(
    decision_node: DecisionNode,
    decision_node_fn=lambda n, d: None,
    chance_node_fn=lambda n, d: None,
    depth=0,
):
    """
    Postorder traversal of the tree rooted at state
    Apply fn once visited
    """
    decision_node_fn(decision_node, depth)

    for chance_node in decision_node.children:
        chance_node_fn(chance_node, depth)
        for next_decision_node in chance_node.children:
            pre_order_traverse(
                next_decision_node, decision_node_fn, chance_node_fn, depth + 1
            )


def get_all_decision_nodes(root: DecisionNode):
    """
    Get all decision nodes in the tree
    """
    decision_nodes = []
    pre_order_traverse(root, decision_node_fn=lambda n, d: decision_nodes.append(n))
    return decision_nodes


def plot_tree(root: DecisionNode, env, filename):
    """
    Plot the tree rooted at root
    """
    tokenizer = env.tokenizer

    def printer(node: ChanceNode, depth):
        # print the average return of the *parent* of this state
        # (this is easier to implement than printing all its children nodes)
        print(
            "\t" * depth,
            repr(tokenizer.decode(node.action)),
            "p",
            node.prob,
            "q",
            chance_node_value(node),
            "len(returns)",
            len(node.sampled_returns),
        )

    pre_order_traverse(root, chance_node_fn=printer)

    # plot the tree
    G = nx.DiGraph()
    G.add_node(root.id, label="<PD>")

    def add_node(node: ChanceNode, depth):
        if len(node.children) > 0:
            child_id = node.children[0].id
            parent_id = node.parent.id

            G.add_node(child_id)
            G.add_edge(parent_id, child_id, label=repr(tokenizer.decode(node.action)))

    pre_order_traverse(root, chance_node_fn=add_node)

    plt.figure(figsize=(15, 15))

    pos = hierarchy_pos(G, root=root.id)
    nx.draw(G, pos, with_labels=True)

    edge_labels = nx.get_edge_attributes(G, "label")
    # plot labels on the edges horizontally
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, label_pos=0.5, rotate=False
    )

    plt.savefig(filename + ".pdf", format="pdf")
    plt.close()


def convert_to_json(root: DecisionNode, env, selected_act):
    """
    Save the information of children of root into a list.
    Does not distinguish layers. So works when the tree only expands one level.
    """
    ret = []

    def get_info(node: ChanceNode, depth):
        if node.action == env.terminal_token:
            # terminal state has no complete_program attribute, since the program is already complete
            complete_program = env.convert_state_to_program(node.children[0].state)
        else:
            complete_program = env.convert_state_to_program(
                node.children[0].info["complete_program"]
            )

        info = {
            "token": env.tokenizer.decode(node.action),
            "state": env.convert_state_to_program(node.children[0].state),
            "selected": node.action == selected_act,
            "score": chance_node_value(node),
            "complete_program": complete_program,
        }
        ret.append(info)

    pre_order_traverse(root, chance_node_fn=get_info)
    return ret
