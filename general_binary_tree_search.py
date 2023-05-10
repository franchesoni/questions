import numpy as np


def get_best_guess(guess_length, indices, predictions):
    # computes the best guess given n.
    assert guess_length > 1
    assert guess_length < len(indices)
    best_guess = [ind for ind in np.argsort(predictions)[::-1] if ind in indices][
        :guess_length
    ]
    return best_guess


# Define the transition function
def transition(guess, indices, predictions):
    # Return a list of possible next states and their probabilities given the current state and action
    assert set(indices).intersection(set(guess)) == set(
        guess
    ), "guess should be a subset of indices"
    pos_state = guess
    neg_state = list(set(indices) - set(guess))
    current_mass = np.sum(predictions[indices])
    pos_prob = np.prod([predictions[i] / current_mass for i in pos_state])
    neg_prob = 1 - pos_prob
    return [(pos_state, pos_prob), (neg_state, neg_prob)]


# Define the reward function
def approx_cost_function(indices, predictions, mode="entropy"):
    if mode == "entropy":
        probs = predictions[indices] / np.sum(predictions[indices])
        return np.sum(-probs * np.log2(probs))
    elif mode == "legnth":
        return np.log2(len(indices))
    else:
        raise ValueError("mode should be either entropy or length")


############# TREE SEARCH ################


# Define the node class
class ActionNode:
    def __init__(self, guess, parent):
        self.guess = guess
        self.parent = parent

        self.state_children = []
        self.children_probs = []
        self.cost = 0
        self.priority = None  # coming from the softmax of the costs

    def add_state_children(self, next_states):
        for state, prob in next_states:
            state_child = StateNode(state, self)
            self.state_children.append(state_child)
            self.children_probs.append(prob)


class StateNode:
    def __init__(self, indices, parent=None):
        # Initialize the node with a state and a parent
        self.indices = indices
        self.parent = parent
        # Initialize the children and the visits as empty lists
        self.action_children = []
        self.softmax_denominator = None
        # Initialize the value as zero
        self.cost = np.inf

    def update_softmax_denominator(self):
        self.softmax_denominator = sum(
            [np.exp(-child.cost) for child in self.action_children]
        )
        return self.softmax_denominator


def update_priorities(state_node):
    smax_denom = state_node.update_softmax_denominator()
    for action_child in state_node.action_children:
        action_child.priority = (
            np.exp(-action_child.cost) / smax_denom * state_node.priority
        )
        for child_ind, state_child in enumerate(action_child.state_children):
            state_child.priority = (
                action_child.children_probs[child_ind] * action_child.priority
            )
            update_priorities(state_child)


def select_node(state_node):
    update_priorities(state_node)
    leaves = get_leaves(state_node)
    non_terminal_leaves = [leaf for leaf in leaves if 1 < len(leaf.indices)]
    if len(non_terminal_leaves) == 0:
        return "all nodes expanded"
    return non_terminal_leaves[
        np.argmax([leaf.priority for leaf in non_terminal_leaves])
    ]


def get_leaves(state_node):
    # get state leaves from a state node
    leaves = []
    for action_child in state_node.action_children:
        for state_child in action_child.state_children:
            leaves.extend(get_leaves(state_child))
    if len(leaves) == 0:
        leaves.append(state_node)
    return leaves


def update_action_cost_depth_0(action_node):
    # update the cost of an action node
    action_node.cost = (
        1  # the cost of taking the action is one question + the expected next cost
    )
    for child_ind, state_child in enumerate(action_node.state_children):
        action_node.cost += state_child.cost * action_node.children_probs[child_ind]


def add_guess(guess, state_node, predictions):
    next_states = transition(guess, state_node.indices, predictions)
    action_child = ActionNode(guess, state_node)
    action_child.add_state_children(next_states)
    state_node.action_children.append(action_child)
    for state_child in action_child.state_children:
        state_child.cost = approx_cost_function(state_child.indices, predictions)
    update_action_cost_depth_0(action_child)
    return action_child


def expand_node(state_node, predictions):
    # expand a state node by considering actions in increasing n and their implied states
    best_cost = state_node.cost
    n = len(state_node.action_children) + 1
    max_n = len(state_node.indices) // 2  # all other are equivalent
    random_is_best = False

    if n == 1:
        support_set = set(state_node.indices)
        sorted_predictions = [
            ind for ind in np.argsort(predictions)[::-1] if ind in support_set
        ]
        # most likely point
        best_guess = [sorted_predictions[0]]
        action_child = add_guess(best_guess, state_node, predictions)
        if action_child.cost < best_cost:
            best_cost = action_child.cost
        # least likely point
        best_guess = [sorted_predictions[-1]]
        action_child = add_guess(best_guess, state_node, predictions)
        if action_child.cost < best_cost:
            best_cost = action_child.cost
        # randomly sampled points
        n_avail = len(sorted_predictions) - 2
        n_samples = min(10, n_avail)
        if n_avail > 0:
            for index in np.random.choice(
                sorted_predictions[1:-1], n_samples, replace=False
            ):
                best_guess = [index]
                action_child = add_guess(best_guess, state_node, predictions)
                if action_child.cost < best_cost:
                    best_cost = action_child.cost
                    random_is_best = True
        n += 1
    while n <= max_n:
        best_guess = get_best_guess(n, state_node.indices, predictions)
        action_child = add_guess(best_guess, state_node, predictions)
        if action_child.cost < best_cost:
            best_cost = action_child.cost
            random_is_best = False
        n += 1
    if random_is_best:
        breakpoint()
    state_node.cost = best_cost


def update_parents(state_node):
    # update parents of state node by going up, computing stop criterion and expanding if needed.
    # Continue upwards if chaning the value function of parent state node
    if state_node.parent is None:
        return
    parent_action = state_node.parent
    update_action_cost_depth_0(parent_action)
    parent_state = parent_action.parent
    if parent_action.cost < parent_state.cost:
        parent_state.cost = parent_action.cost
        update_parents(
            parent_state,
        )  # assume they're up to date unless parent_state.cost changes


def get_best_action(state_node):
    # get the best action from a state node
    return state_node.action_children[
        np.argmin([action_child.cost for action_child in state_node.action_children])
    ]


def get_node_with_indices(root_node, indices):
    """Explores the tree and returns the node with the given state using breadth first"""
    queue = [root_node]
    while len(queue) > 0:
        node = queue.pop(0)
        if node.indices == indices:
            return node
        for action_child in node.action_children:
            queue.extend(action_child.state_children)
    return None


def initialize_tree(root_state):
    root_node = StateNode(root_state)
    root_node.priority = 1  # trickle down priorities
    return root_node


def tree_search(root_node, max_expansions, predictions):
    """Here the root_node is a set of possible labelings and predictions is the probability of each one of the possible labelings."""

    # print("Expanding tree...")
    # for i in tqdm.tqdm(range(max_expansions)):
    for i in range(max_expansions):
        node = select_node(root_node)  # node with lowest entropy
        if node == "all nodes expanded":
            break
        expand_node(node, predictions)
        update_parents(node)

    best_action = get_best_action(root_node)
    return best_action.guess, best_action.cost, best_action.children_probs[0]
