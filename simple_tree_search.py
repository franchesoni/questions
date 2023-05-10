import numpy as np
import tqdm
import gc

"""
Our state is a list of indices and a list of probabilities and the last and only informative incorrect guess.
This is, state = {"indices": unlabeled_indices, "last_incorrect_guess": (indices, labels)}
part of the state are the current predictions, which are (indices, probs)
The action has as guess a list of indices and a list of labels. This is, action.guess = (indices, labels)
The labels should be deductible from "probs" but we save them for the case of incorrect guesses where we can deduce one more labeling for free.
"""


def entropy_given_state_preds_binary(state, predictions):
    assert predictions[0] == state["indices"]
    probabilities = np.array(
        [pred for ind, pred in zip(*predictions) if ind in state["indices"]]
    )
    H_Y = entropy_given_probs_binary(probabilities)
    if state["incorrect"] is None:
        return H_Y
    # else if there is an incorrect guess we compute conditional entropy
    # H(Y|x wrong) = -sum_y p(y| x wrong) log p(y|x wrong)
    # now note that p(y|x wrong) = p(x wrong | y) p(y) / p(x wrong) and that p(x wrong | y) = p(y)/p(x wrong) if x is not included in y
    # then H(Y|x wrong) = -sum_y\x p(y)/p(x w) log (p(y)/p(x w))
    # = 1/p(x w) * [H(Y) - H(Y outside x) + p(x w)log p(x w)]
    # = 1/(1-p(x)) * [H(Y) - H(Y outside x) + (1-p(x))log (1-p(x))]
    p_x = np.prod([prob for ind, prob in zip(*predictions) if ind in state["incorrect"][0]])
    p_x_w = 1 - p_x
    probs_Y_outside_x = np.array([pred for ind, pred in zip(*predictions) if ind not in state["incorrect"][0]])
    H_Y_outside_x = entropy_given_probs_binary(probs_Y_outside_x)
    H_Y_given_x_wrong = 1/p_x_w * (H_Y - H_Y_outside_x + p_x_w * np.log2(p_x_w))
    return H_Y_given_x_wrong


def entropy_given_probs_binary(probabilities):
    return -np.sum(
        probabilities * np.log2(probabilities)
        + (1 - probabilities) * np.log2(1 - probabilities)
    )


def log2_number_of_possible_labelings(state, predictions):
    N = len(state["indices"])
    if state["incorrect"] is not None:
        # if there is an incorrect guess we discard 2^(N-inc) labelings where inc is the number of indices in the incorrect guess
        inc = len(state["incorrect"][0])
        ret = N - inc + np.log2(2**inc - 1)
        return ret
    else:
        return N


class StateNode:
    def __init__(self, state, parent=None):
        # Initialize the node with a state and a parent
        self.state = state
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


class STS:
    def __init__(self, max_expansions, al_method=None, max_n=10, cost_fn="entropy"):
        self.max_expansions = max_expansions
        self.predictions_changed = False
        self.al_method = al_method
        self.max_n = max_n
        self.approx_cost_function = (
            entropy_given_state_preds_binary
            if cost_fn == "entropy"
            else log2_number_of_possible_labelings
        )

    def set_unlabeled_predictions(self, predictions):
        self.predictions = predictions
        self.predictions_changed = True

    # all other methods are class methods, we prefer a functional approach and don't use state

    def get_root_node(root_state):
        root_node = StateNode(root_state)
        root_node.priority = 1
        if root_node.parent:
            STS._destroy_upwards(root_node.parent)
        return root_node

    # define decorator that sets self.predictions_changed to False
    def require_new_predictions(func):
        def wrapper(self, *args, **kwargs):
            if not self.predictions_changed:
                raise ValueError("You need to set the predictions first!")
            ret = func(self, *args, **kwargs)
            self.predictions_changed = True
            return ret

        return wrapper

    @require_new_predictions
    def tree_search(self, root_node):
        """Here the root_node is a set of possible labelings and predictions is the probability of each one of the possible labelings."""
        predictions = self.predictions  # (indices, probs)
        al_method = self.al_method
        max_n = self.max_n
        approx_cost_function = self.approx_cost_function

        # print("Expanding tree...")
        # for i in tqdm.tqdm(range(self.max_expansions)):
        for i in range(self.max_expansions):
            node = STS._select_node(root_node)  # node with lowest entropy
            if node == "all nodes expanded":
                break
            node_predictions = (
                node.state["indices"],
                [
                    predictions[1][predictions[0].index(ind)]
                    for ind in node.state["indices"]
                ],
            )
            STS._expand_node(
                node, max_n, node_predictions, al_method, approx_cost_function
            )
            STS._update_parents(node)  # here

        best_action = STS._get_best_action(root_node)
        assert len(best_action.guess[1]) > 0
        return best_action.guess, best_action.cost, best_action.children_probs[0]

    def _destroy_upwards(node):
        if node.parent:
            STS._destroy_upwards(node.parent)
            node.parent = None
        del node
        gc.collect()

    def _select_node(state_node):
        STS._update_priorities(state_node)
        leaves = STS._get_leaves(state_node)
        non_terminal_leaves = [
            leaf for leaf in leaves if 0 < len(leaf.state["indices"])
        ]
        if len(non_terminal_leaves) == 0:
            return "all nodes expanded"
        return non_terminal_leaves[np.argmax([leaf.priority for leaf in non_terminal_leaves])]

    def _update_priorities(state_node):
        smax_denom = state_node.update_softmax_denominator()
        for action_child in state_node.action_children:
            action_child.priority = (
                np.exp(-action_child.cost) / smax_denom * state_node.priority
            )
            for child_ind, state_child in enumerate(action_child.state_children):
                state_child.priority = (
                    action_child.children_probs[child_ind] * action_child.priority
                )
                STS._update_priorities(state_child)

    def _get_leaves(state_node):
        # get state leaves from a state node
        leaves = []
        for action_child in state_node.action_children:
            for state_child in action_child.state_children:
                leaves.extend(STS._get_leaves(state_child))
        if len(leaves) == 0:
            leaves.append(state_node)
        return leaves

    def _expand_node(node, max_n, predictions, al_method, approx_cost_function):
        # expand a state node by considering actions in increasing n and their implied states
        assert node.state['indices'] == predictions[0]
        best_cost = node.cost
        n = len(node.action_children) + 1
        assert n <= max_n, "you want to add an action but you've already added them all"
        max_n = min(max_n, len(predictions[0]))
        if node.state["incorrect"] is not None:
            assert len(node.state['incorrect'][0]) > 1
            # we know the next guess
            # this is the only action we will consider
            inc_indices, inc_labels = node.state["incorrect"]
            certainties = np.abs(
                np.array(predictions[1])[
                    [predictions[0].index(ind) for ind in inc_indices]
                ]
                - 0.5
            )
            less_certain_index = np.argsort(certainties)[0]
            new_indices = (
                inc_indices[:less_certain_index] + inc_indices[less_certain_index + 1 :]
            )
            new_labels = (
                inc_labels[:less_certain_index] + inc_labels[less_certain_index + 1 :]
            )
            best_guess = (new_indices, new_labels)
            action_child = STS._add_guess(
                best_guess, node, predictions, approx_cost_function
            )
            if action_child.cost < best_cost:
                best_cost = action_child.cost
                assert len(action_child.guess[0]) > 0
        else:
            if n == 1 and al_method is None:  # get most and least certain points
                sorted_predictions_indices = np.argsort(np.abs(np.array(predictions[1])-0.5))[::-1]
                # most likely point
                for best_guess in [
                    ([predictions[0][sorted_predictions_indices[0]]], [0]),
                    ([predictions[0][sorted_predictions_indices[-1]]], [0]),
                ]:
                    action_child = STS._add_guess(
                        best_guess, node, predictions, approx_cost_function
                    )
                    if action_child.cost < best_cost:
                        best_cost = action_child.cost
                    if len(predictions) == 1:  # last prediction
                        node.cost = best_cost
                        assert len(action_child.guess[0]) > 0
                        return
                n += 1
            while n <= max_n:
                best_guess = STS._get_best_guess(n, node.state, predictions)
                action_child = STS._add_guess(
                    best_guess, node, predictions, approx_cost_function
                )
                if action_child.cost < best_cost:
                    best_cost = action_child.cost
                n += 1
        assert len(action_child.guess[0]) > 0
        node.cost = best_cost

    def _add_guess(guess, state_node, predictions, approx_cost_function):
        next_states = STS._transition(guess, state_node.state, predictions)
        action_child = ActionNode(guess, state_node)
        for ns in [trans[0] for trans in next_states]:
            assert ns['incorrect'] is None or len(ns["incorrect"][0]) > 1
        action_child.add_state_children(next_states)
        state_node.action_children.append(action_child)
        for state_child in action_child.state_children:
            child_predictions = (state_child.state['indices'], [pred for ind, pred in zip(*predictions) if ind in state_child.state['indices']])
            state_child.cost = approx_cost_function(state_child.state, child_predictions)
        STS._update_action_cost_depth_0(action_child)
        return action_child

    # Define the transition function
    def _transition(guess, state, predictions):
        # Return a list of possible next states and their probabilities given the current state and action
        assert set(guess[0]).intersection(set(state["indices"])) == set(
            guess[0]
        ), "guess should be a subset of indices"
        pos_state = STS.update_state(state, guess, True)
        neg_state = STS.update_state(state, guess, False)
        pos_prob = STS._compute_prob(state, guess, predictions)
        neg_prob = 1 - pos_prob
        return [(pos_state, pos_prob), (neg_state, neg_prob)]

    def update_state(state, guess, is_correct):
        """Updates the state {'indices', 'incorrect'} with guess"""
        indices, incorrect = state["indices"], state["incorrect"]
        if len(guess[0]) == 1 and not is_correct:  # a single digit is always right
            return STS.update_state(state, (guess[0], [1 - guess[1][0]]), True)
        if incorrect is not None and is_correct:
            assert STS._first_is_shortened_second(
                guess[0], incorrect[0]
            ), "incorrect should be a child of the previous incorrect"
            # incorrect is a child of previous incorrect
            # the wrong digit was in previous incorrect
            inc_ind = list(set(incorrect[0]) - set(guess[0]))[0]
            # now we remove all the indices of guess and the inc_ind from the state (we don't care about the label here but only on the outer loop. We care just about the next state)
            to_remove = guess[0] + [inc_ind]
            new_indices = [ind for ind in indices if ind not in to_remove]
            new_incorrect = None
        elif incorrect is None and is_correct:
            new_indices = [ind for ind in indices if ind not in guess[0]]
            new_incorrect = None
        elif not is_correct:
            # we remove the previous incorrect and add the guess as incorrect
            new_indices = indices
            new_incorrect = guess
        return {"indices": new_indices, "incorrect": new_incorrect}

    def _first_is_shortened_second(first, second):
        """Checks if the first guess is the second guess without an element"""
        return len(set(second) - set(first)) == 1

    def _compute_prob(state, guess, predictions):
        assert state["indices"] == predictions[0]
        pg = STS._prob_of_guess(guess, predictions)
        if state["incorrect"] is None:
            # clean state, the probability is the probability of the guess
            return pg
        # else there is an incorrect state
        # the incorrect state is a parent of the guess
        # p(g ok |inc wrong) = (1-p(inc\g ok))p(g ok) / (1-p(g ok)p(inc\g ok))
        # then the only thing we have to compute is the probability of the extra digit j of the previous incorrect guess
        inc = state["incorrect"]
        extra_index = list(set(inc[0]) - set(guess[0]))[0]
        p_j = predictions[1][predictions[0].index(extra_index)]
        prob = (1 - p_j) * pg / (1 - pg * p_j)
        return prob

    def _prob_of_guess(guess, predictions):
        prob = 1
        for i, label in zip(guess[0], guess[1]):
            ind_of_predictions = predictions[0].index(i)
            symbol_prob = predictions[1][ind_of_predictions]
            prob *= symbol_prob**label * (1 - symbol_prob) ** (1 - label)
        return prob

    def _update_action_cost_depth_0(action_node):
        # update the cost of an action node
        action_node.cost = (
            1  # the cost of taking the action is one question + the expected next cost
        )
        for child_ind, state_child in enumerate(action_node.state_children):
            action_node.cost += state_child.cost * action_node.children_probs[child_ind]

    def _get_best_guess(guess_length, state, predictions):
        # computes the best guess given n.
        assert guess_length > 1
        assert state["incorrect"] is None
        indices = state["indices"]
        assert guess_length <= len(indices)
        assert (
            state["indices"] == predictions[0]
        ), "state and predictions should have the same indices"
        # the best guess is given by the n most certain predictions
        certainty = np.abs(
            np.array(predictions[1]) - 0.5
        )  # the closer to 0.5 the less certain
        ordered_values_indices = np.argsort(certainty)[::-1].tolist()
        chosen_values_indices = ordered_values_indices[:guess_length]
        best_guess = (
            (np.array(indices)[chosen_values_indices]).tolist(),
            np.round(np.array(predictions[1])[chosen_values_indices]).tolist(),
        )
        return best_guess

    def _update_parents(state_node):
        # update parents of state node by going up, computing stop criterion and expanding if needed.
        # Continue upwards if chaning the value function of parent state node
        if state_node.parent is None:
            return
        parent_action = state_node.parent
        STS._update_action_cost_depth_0(parent_action)
        parent_state = parent_action.parent
        if parent_action.cost < parent_state.cost:
            parent_state.cost = parent_action.cost
            STS._update_parents(
                parent_state,
            )  # assume they're up to date unless parent_state.cost changes

    def _get_best_action(state_node):
        # get the best action from a state node
        return state_node.action_children[
            np.argmin(
                [action_child.cost for action_child in state_node.action_children]
            )
        ]
