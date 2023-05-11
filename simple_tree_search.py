import time
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
TEMPERATURE = 10  # more than 1
MAX_DEPTH = 20  # that's already a lot
PROB_BUFFER_FACTOR = 0.01

def entropy_given_whats_wrong(probabilities, indices_Y):
    # consider the random variable X given by binary vectors of length N
    # for this random variable, we can compute the entropy as follows:
    # H(X) = -sum_{x in X} P(X=x)log(P(X=x))
    # where sum_{x in X} P(x) = 1 because it's a probability distribution
    # if now we're told that x is not in a set Y, we have to update our P(X=x)
    # P(X=x|x not in Y) = P(x not in Y|x)P(x)/P(x not in Y)
    # = (1 - P(x in Y|x))P(x) / (1 - P(x in Y))
    probabilities = np.array(probabilities)
    if len(indices_Y) > 0:
        p_x_given_Y = []
        for ind, p_x in enumerate(probabilities):
            p_x_not_in_Y_given_x = 1 - 1*(ind in indices_Y)
            p_x_in_Y = np.sum(probabilities[indices_Y])
            p_x_given_Y.append(p_x_not_in_Y_given_x * p_x / (1 - p_x_in_Y))
        probabilities = np.array(p_x_given_Y)
    is_zero = probabilities == 0
    entropies = -probabilities * np.log2(probabilities)
    entropies[is_zero] = 0
    return np.sum(entropies)

def get_power_01(size):
    """Returns all possible binary vectors of size `size`."""
    return [[int(d) for d in np.binary_repr(i, width=size)] for i in range(2**size)]

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
    # now note that p(y|x wrong) = p(x wrong | y) p(y) / p(x wrong) and that p(x wrong | y) = 1 if they're inconsistent and 0 otherwise
    # then H(Y|x wrong) = -sum_{y_inconsistent_x} p(y)/p(x w) log (p(y)/p(x w))
    # now we need to consider that y_inconsistent_x are a lot of binary vectors, we need to use tricks
    # first, x_w doesn't say anything about the distribution outside x_w
    # H(Y|x w) = H(Y outside x) + H(Y at x_w|x_w)
    probs_Y_outside_x = np.array(
        [pred for ind, pred in zip(*predictions) if ind not in state["incorrect"][0]]
    )
    H_Y_outside_x = entropy_given_probs_binary(probs_Y_outside_x)
    # now we need to compute H(Y at x_w|x_w) which is the sum of many entropies (we compute the full thing now, not based on binary)
    p_x_w = 1 - STS._prob_of_guess(state['incorrect'], predictions)  # p(x w) = 1 - p(x ok)
    overlapping_predictions = [predictions[1][predictions[0].index(ind)] for ind in state["incorrect"][0]]
    binary_vectors = get_power_01(len(state["incorrect"][0]))
    bin_vec_probs = STS._prob_of_guesses(binary_vectors, overlapping_predictions) 
    index_to_exclude = [ind for ind in range(len(binary_vectors)) if binary_vectors[ind] == state['incorrect'][1]]
    assert len(index_to_exclude) == 1
    H_Y_at_x_given_x_w = entropy_given_whats_wrong(bin_vec_probs, index_to_exclude)
    return H_Y_outside_x + H_Y_at_x_given_x_w


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
    def __init__(self, state, parent=None, depth=0):
        # Initialize the node with a state and a parent
        self.state = state
        self.parent = parent
        self.depth = depth
        # Initialize the children and the visits as empty lists
        self.action_children = []
        self.softmax_denominator = None
        # Initialize the value as zero
        self.cost = np.inf

    def update_softmax_denominator(self):
        self.softmax_denominator = sum(
            [np.exp(-child.cost / TEMPERATURE) for child in self.action_children]
        )
        return self.softmax_denominator


# Define the node class
class ActionNode:
    def __init__(self, guess, parent):
        self.guess = guess
        self.parent = parent
        self.depth = self.parent.depth

        self.state_children = []
        self.children_probs = []
        self.cost = 0
        self.priority = None  # coming from the softmax of the costs

    def add_state_children(self, next_states):
        for state, prob in next_states:
            state_child = StateNode(state, self, self.depth+1)
            self.state_children.append(state_child)
            self.children_probs.append(prob)


class STS:
    def __init__(self, max_expansions, al_method="uncertainty", max_n=10, cost_fn="entropy"):
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
        probs = np.array(predictions[1])
        probs = (probs - PROB_BUFFER_FACTOR * (probs - 0.5)).tolist()
        self.predictions = (predictions[0], probs)
        self.predictions_changed = True

    # all other methods are class methods, we prefer a functional approach and don't use state

    def initialize_root_node(root_state):
        root_node = StateNode(root_state, None, 0)
        return STS.set_new_root_node(root_node)

    def set_new_root_node(root_node):
        root_node.priority = 1
        if root_node.parent:
            STS._destroy_ancestors(root_node)
        gc.collect()  # totally arbitrary, might be useful
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

        best_action_node = STS._get_best_action(root_node)
        assert len(best_action_node.guess[1]) > 0
        return best_action_node 

    def _destroy_ancestors(node):
        if node.parent:
            STS._destroy_ancestors(node.parent)
            del node.parent
            node.parent = None
        gc.collect()

    def _select_node(state_node):
        STS._update_priorities(state_node)
        leaves = STS._get_leaves(state_node)
        non_terminal_leaves = [
            leaf for leaf in leaves if 0 < len(leaf.state["indices"])
        ]
        if len(non_terminal_leaves) == 0:
            return "all nodes expanded"
        return non_terminal_leaves[
            np.argmax([leaf.priority for leaf in non_terminal_leaves])
        ]

    def _update_priorities_recursive(state_node: StateNode):
        smax_denom = state_node.update_softmax_denominator()
        for action_child in state_node.action_children:
            action_child.priority = (
                np.exp(-action_child.cost / TEMPERATURE) / smax_denom * state_node.priority
            )
            for child_ind, state_child in enumerate(action_child.state_children):
                state_child.priority = (
                    action_child.children_probs[child_ind] * action_child.priority
                )
                STS._update_priorities(state_child)

    def _update_priorities(state_node):
        """Use this implementation to avoid recursion depth errors. They happen when the tree is too deep (before 1000).
        We should limit the depth at some point
        """
        stack = [state_node] # create a stack and push the initial node
        while stack: # loop until the stack is empty
            node = stack.pop() # pop the top node from the stack
            smax_denom = node.update_softmax_denominator() # update the softmax denominator
            for action_child in node.action_children: # loop through the action children
                action_child.priority = (
                    np.exp(-action_child.cost / TEMPERATURE) / smax_denom * node.priority
                ) # update the action child priority
                for child_ind, state_child in enumerate(action_child.state_children): # loop through the state children
                    state_child.priority = (
                        action_child.children_probs[child_ind] * action_child.priority
                    ) # update the state child priority
                    stack.append(state_child) # push the state child to the stack


    def _get_leaves(state_node):
        # get state leaves from a state node
        initial_depth = state_node.depth # get the initial depth
        stack = [state_node] # create a stack and push the initial node
        leaves = [] # create an empty list to store the leaves
        while stack: # loop until the stack is empty
            node = stack.pop() # pop the top node from the stack
            if not node.action_children: # check if the node has no action children
                leaves.append(node) # add the node to the list of leaves
                continue # skip the rest of the loop
            for action_child in node.action_children: # loop through the action children
                for state_child in action_child.state_children: # loop through the state children
                    if state_child.depth - initial_depth < MAX_DEPTH:
                        stack.append(state_child) # push the state child to the stack
        return leaves # return the list of leaves



    def _expand_node(node, max_n, predictions, al_method, approx_cost_function):
        # expand a state node by considering actions in increasing n and their implied states
        assert node.state["indices"] == predictions[0]
        original_cost = node.cost  # this was computed at depth=0, we will update it with depth=1
        if len(node.action_children) == 0:
            best_cost = np.inf
        else:
            best_cost = node.cost
        n = len(node.action_children) + 1
        assert n <= max_n, "you want to add an action but you've already added them all"
        max_n = min(max_n, len(predictions[0]))
        if node.state["incorrect"] is not None:
            assert len(node.action_children) == 0, "this should be the first and only action added"
            assert len(node.state["incorrect"][0]) > 1
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
            assert len(action_child.guess[0]) > 0
            assert best_cost == np.inf, "this should be the only one as you have only one option you haven't yet tried"
            node.cost = action_child.cost
            return
        else:
            if len(predictions[0]) == 1:
                assert n == 1
                best_guess = (predictions[0], [0])
                action_child = STS._add_guess(
                    best_guess, node, predictions, approx_cost_function
                )
                assert best_cost == np.inf, "this should be the only one as you have only one option you haven't yet tried"
                assert action_child.cost == 1
                node.cost = action_child.cost
                return
            elif n == 1:
                assert best_cost == np.inf
                if al_method == "uncertainty":  # get most and least certain points
                    sorted_predictions_indices = np.argsort(
                        np.abs(np.array(predictions[1]) - 0.5)
                    )[::-1]
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
                elif al_method is "random":
                    best_guess = (np.random.choice(predictions[0], 1).tolist(), [0])
                    action_child = STS._add_guess(
                        best_guess, node, predictions, approx_cost_function
                    )
                    if action_child.cost < best_cost:  # this should always be true
                        best_cost = action_child.cost
                    else:
                        raise RuntimeError("you shouln'd be here, check best cost")
                else:
                    raise NotImplementedError(
                        "you should have specified a valid active learning method"
                    )
                n += 1
            while n <= max_n:
                assert len(predictions[0]) > 1
                best_guess = STS._get_best_guess(n, node.state, predictions)
                action_child = STS._add_guess(
                    best_guess, node, predictions, approx_cost_function
                )
                assert best_cost < np.inf, "we should have found at least one action by now"
                if action_child.cost < best_cost:
                    best_cost = action_child.cost
                n += 1
        assert len(action_child.guess[0]) > 0
        node.cost = best_cost

    def _add_guess(guess, state_node, predictions, approx_cost_function):
        next_states = STS._transition(guess, state_node.state, predictions)
        action_child = ActionNode(guess, state_node)
        for ns in [trans[0] for trans in next_states]:
            assert ns["incorrect"] is None or len(ns["incorrect"][0]) > 1
        action_child.add_state_children(next_states)
        state_node.action_children.append(action_child)
        for state_child in action_child.state_children:
            child_predictions = (
                state_child.state["indices"],
                [
                    pred
                    for ind, pred in zip(*predictions)
                    if ind in state_child.state["indices"]
                ],
            )
            state_child.cost = approx_cost_function(
                state_child.state, child_predictions
            )
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
        return [(neg_state, neg_prob), (pos_state, pos_prob)]

    def update_state(state, guess, is_correct, ret_to_remove=False):
        """Updates the state {'indices', 'incorrect'} with guess"""
        unlabeled_indices, incorrect = state["indices"], state["incorrect"]
        inverted = False
        if len(guess[0]) == 1 and not is_correct:  # a single digit is always right
            question = (guess[0], [1 - guess[1][0]])
            is_correct = True
            inverted = True  # we have inverted the guess
        if incorrect is not None and is_correct:
            assert STS._first_is_shortened_second(
                guess[0], incorrect[0]
            ), "incorrect should be a child of the previous incorrect"
            # guess should be a child of incorrect if not inverted
            if inverted:
                to_remove = guess[0]
            else:  # guess wasn't inverted, then we know that the extra bit in incorrect was wrong and we annotate it correctly
                inc_ind = list(set(incorrect[0]) - set(guess[0]))[0]
                to_remove = guess[0] + [inc_ind]
            # now we remove all the indices of guess and the inc_ind from the state (we don't care about the label here but only on the outer loop. We care just about the next state)
            new_indices = [ind for ind in unlabeled_indices if ind not in to_remove]
            new_incorrect = None
        elif incorrect is None and is_correct:
            to_remove = guess[0]
            new_indices = [ind for ind in unlabeled_indices if ind not in to_remove]
            new_incorrect = None
        elif not is_correct:
            # we remove the previous incorrect and add the guess as incorrect
            to_remove = []  # this is useful when controlling from outside
            new_indices = unlabeled_indices
            new_incorrect = guess
        if ret_to_remove:
            return {"indices": new_indices, "incorrect": new_incorrect}, to_remove
        return {"indices": new_indices, "incorrect": new_incorrect}
        

    def _first_is_shortened_second(first, second):
        """Checks if the first guess is the second guess without an element"""
        return len(set(second) - set(first)) == 1

    def _first_is_child_of_second(first, second):
        ind1, lab1 = first
        ind2, lab2 = second
        # first is a child if doesn't disagree with second
        for firstind, firstlabel in zip(*first):
            if (not firstind in ind2) or (lab2[ind2.index(firstind)] != firstlabel):
                return False
        return True

    def _are_inconsistent(first, second):
        """Two guesses are inconsistent if they disagree on the shared support"""
        shared_indices = list(set(first[0]).intersection(set(second[0])))
        for ind in shared_indices:
            if first[1][first[0].index(ind)] != second[1][second[0].index(ind)]:
                return True
        return False

    def _compute_prob(state, guess, predictions):
        assert state["indices"] == predictions[0]
        pg = STS._prob_of_guess(guess, predictions)
        if state["incorrect"] is None:
            # clean state, the probability is the probability of the guess
            return pg
        # else there is an incorrect state
        # p(g ok |inc wrong) = p(inc w | g ok)p(g ok) / p(inc w) = (1-p(inc ok|g ok))p(g ok) / (1-p(inc ok))
        inc = state["incorrect"]
        p_inc_ok = STS._prob_of_guess(inc, predictions)
        # guess is not a parent, then it differs with incorrect in one lacking digit (is a child) or is inconsistent
        # p(inc ok | g ok) = 0 if they are inconsistent
        # p(inc ok | g ok) = p(d) otherwise, which is the probability of the extra digits d of inc being correct
        # then the only thing we have left to compute is the probability of the extra digit j of the previous incorrect guess
        if STS._are_inconsistent(inc, guess):
            p_inc_ok_given_g_ok = 0
        else:  # there should be an extra digit
            inc_ind = [ind for ind in inc[0] if ind not in guess[0]]
            inc_lab = [lab for ind,lab in zip(*inc) if ind not in guess[0]]
            p_d = STS._prob_of_guess((inc_ind, inc_lab), predictions)
            p_inc_ok_given_g_ok = p_d
        prob = (1 - p_inc_ok_given_g_ok) * pg / (1 - p_inc_ok)
        if prob < 0:
            assert np.allclose(prob, 0)
            return 0
        if prob > 1:
            assert np.allclose(prob, 1)
            return 1
        return prob

    def _prob_of_guess(guess, predictions):
        prob = 1
        for i, label in zip(guess[0], guess[1]):
            ind_of_predictions = predictions[0].index(i)
            symbol_prob = predictions[1][ind_of_predictions]
            prob *= symbol_prob**label * (1 - symbol_prob) ** (1 - label)
        return prob

    def _prob_of_guesses(guesses, predictions):
        # now guesses has shape (B, n) and predictions has shape (n) and they are aligned
        guesses, predictions = np.array(guesses), np.array(predictions).reshape(1, -1)
        probs = predictions ** guesses * (1 - predictions) ** (1 - guesses)  # (B, n)
        return np.prod(probs, axis=1).tolist()  # (B)


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




######## TESTS ########
def sample_incorrect(inc_guess_len, indices):
    incorrect_indices = np.random.choice(indices, inc_guess_len, replace=False).tolist()
    incorrect_guess = np.random.randint(2, size=inc_guess_len).tolist()
    incorrect = (incorrect_indices, incorrect_guess)
    return incorrect

def sample_ground_truth(probabilities, incorrect):
    if incorrect is not None:
        incorrect_indices, incorrect_guess = incorrect
    st = time.time()
    while True:
        ground_truth = (np.random.rand(len(probabilities)) < np.array(probabilities[1])).astype(int).tolist()
        if incorrect is None:
            break
        if not (np.array(ground_truth)[incorrect_indices] == np.array(incorrect_guess)).all():
            break  # the incorrect guess is not equal to the ground truth
        if time.time() - st > 1:
            breakpoint()
            raise TimeoutError("Couldn't find a ground truth in 1s")
    return ground_truth

def sample_guess(indices, incorrect):
    if incorrect is None:
        guess_length = np.random.choice(np.arange(1, len(indices)+1))
        guess_indices = np.random.choice(indices, guess_length, replace=False).tolist()
        guess_labels = np.random.randint(2, size=len(indices))[guess_indices].tolist()
        return (guess_indices, guess_labels)
    else:
        while True:  # can't be a parent of incorrect
            guess_indices, guess_labels = sample_guess(indices, None)
            for ind, inclabel in zip(*incorrect):
                if (not ind in guess_indices) or (guess_labels[guess_indices.index(ind)] != inclabel):
                    return (guess_indices, guess_labels)

def sample_probs(indices):
    probs = np.random.randint(1, 100, size=len(indices))/100
    probs = probs / probs.sum()
    return (indices, probs.tolist())




def test_probability_computation(runs=100000):
    """We will test compute_prob. Our state are indices [0,1,2], one random incorrect guess (of size at least 2) and some probabilities."""
    np.random.seed(0)
    indices = [0, 1, 2]
    state = {'indices': indices, 'incorrect': None}
    probs_and_answers = {'none': [], '2': [], '3': []}
    for tryn in tqdm.tqdm(range(runs)):
        probabilities = sample_probs(indices)
        guess = sample_guess(indices, None)
        ground_truth = sample_ground_truth(probabilities, None)
        prob_pos = STS._compute_prob(state, guess, probabilities)
        answer = [ground_truth[indices.index(ind)] for ind in guess[0]] == guess[1]
        probs_and_answers['none'].append((prob_pos, answer))

    for inc_guess_len in [2, 3]:
        for tryn in tqdm.tqdm(range(runs)):
            incorrect = sample_incorrect(inc_guess_len, indices)
            state['incorrect'] = incorrect
            probabilities = sample_probs(indices)
            guess = sample_guess(indices, incorrect)
            if [probabilities[1][probabilities[0].index(ind)] for ind in incorrect[0]] == incorrect[1]:
                continue
            ground_truth = sample_ground_truth(probabilities, incorrect) 
            prob_pos = STS._compute_prob(state, guess, probabilities)
            answer = [ground_truth[indices.index(ind)] for ind in guess[0]] == guess[1]
            probs_and_answers[str(inc_guess_len)].append((prob_pos, answer))

    from visualization import calibration_plot2
    calibration_plot2(probs_and_answers, 11, 'testSTS')




def test_entropy_computation(runs=100000):
    """Here we test the entropy computation of a state by checking against the definition."""
    from huffman import get_power_01
    np.random.seed(0)
    indices = [0, 1, 2]
    binary_vectors = get_power_01(len(indices))
    state = {'indices': indices, 'incorrect': None}
    entropies_and_gold = {'none': [], '2': [], '3': []}
    for tryn in tqdm.tqdm(range(runs)):
        predictions = sample_probs(indices)
        bin_vec_probs = [STS._prob_of_guess((indices, bin_vec), predictions) for bin_vec in binary_vectors]
        gold_entropy = entropy_given_whats_wrong(bin_vec_probs, [])
        entropy = entropy_given_state_preds_binary(state, predictions)
        entropies_and_gold['none'].append((entropy, gold_entropy))

    for inc_guess_len in [2, 3]:
        for tryn in tqdm.tqdm(range(runs)):
            incorrect = sample_incorrect(inc_guess_len, indices)
            state['incorrect'] = incorrect
            predictions = sample_probs(indices)
            bin_vec_probs = [STS._prob_of_guess((indices, bin_vec), predictions) for bin_vec in binary_vectors]
            # now we need to obtain the indices of those binary vectors that are tagged as incorrect by the incorrect guess
            # these are all the parents of the incorrect guess and the incorrect guess itself
            to_remove_indices = [ind for ind, bin_vec in enumerate(binary_vectors) if STS._first_is_child_of_second(incorrect, (indices, bin_vec))]
            gold_entropy = entropy_given_whats_wrong(bin_vec_probs, to_remove_indices)
            entropy = entropy_given_state_preds_binary(state, predictions)
            entropies_and_gold[str(inc_guess_len)].append((entropy, gold_entropy))

    from visualization import entropies_and_gold as plot
    plot(entropies_and_gold, 'testSTS_entropy')



def run_tests():
    # import cProfile
    # cProfile.run('test_probability_computation()', sort='cumtime', filename='testSTS.profile')
    test_entropy_computation(1000)

run_tests()