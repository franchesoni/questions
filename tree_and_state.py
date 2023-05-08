import tqdm
import itertools
import numpy as np
from copy import deepcopy


##### STATE HANDLING #####
def get_power_01(size):
    """Returns all possible binary vectors of size `size`."""
    return [[int(d) for d in np.binary_repr(i, width=size)] for i in range(2**size)]



def are_consistent_inc_cor(incorrect_guesses, correct_guess):
    """Checks if the incorrect guesses are consistent with the correct guess.
    This is, no incorrect guess can be included in the correct guess."""
    cor_indices, cor_labels = correct_guess
    for incorrect_guess in incorrect_guesses:
        inc_indices, inc_labels = incorrect_guess
        is_included = True
        for iind, ilabel in zip(inc_indices, inc_labels):
            if iind not in set(cor_indices) or ilabel != cor_labels[cor_indices.index(iind)]:
                is_included = False
                break
        if is_included:
            return False
    return True


def are_consistent_cor_cor(guess1, guess2):
    """Checks if the two correct guesses are consistent."""
    ind1, lab1 = guess1
    ind2, lab2 = guess2
    for i in range(len(ind1)):
        if ind1[i] in set(ind2):
            if lab1[i] != lab2[ind2.index(ind1[i])]:
                return False
    return True




def take_differing_bits(guess, ref_guess):
    """Removes those indices of guess that are in ref_guess.
    Assume consistency_cor_cor."""
    indices, labels = guess
    ref_indices, ref_labels = ref_guess
    new_indices = []
    new_labels = []
    for i in range(len(indices)):
        if indices[i] not in set(ref_indices):
            new_indices.append(indices[i])
            new_labels.append(labels[i])
    return new_indices, new_labels


def remove_differing_bits(old_guesses, new_guess):
    """e have a list of old guesses where each guess is a tuple `(indices, labels)` and a new guess which is itself a tuple `(indices, labels)`. `indices` and `labels` are always list of ints and each label is either 0 or 1. When the new guess has the same indices than an old guess and differs with the old one by exactly one label, then the differing label and its corresponding indices should be removed from the old guess."""
    new_indices, new_labels = new_guess
    for i, (old_indices, old_labels) in enumerate(old_guesses):
        if len(new_indices) == len(old_indices) and set(new_indices) == set(
            old_indices
        ):
            differing_indices = []
            differing_labels = []
            for j in range(len(new_indices)):
                if new_labels[j] != old_labels[old_indices.index(new_indices[j])]:
                    differing_indices.append(new_indices[j])
                    differing_labels.append(new_labels[j])
            if len(differing_indices) == 1:
                diff_ind = old_indices.index(differing_indices[0])
                old_labels.pop(diff_ind)
                old_indices.pop(diff_ind)
    return old_guesses

def are_parent_and_child(parent_guess, child_guess):
    """Checks if the parent contains the child by containing the same indices and labels."""
    parent_inds, parent_labels = parent_guess
    child_indices, child_labels = child_guess
    parent_inds_set = set(parent_inds)
    for cind in child_indices:
        if not cind in parent_inds_set:
            return False
    for cind, clabel in zip(child_indices, child_labels):
        if clabel != parent_labels[parent_inds.index(cind)]:
            return False
    return True


def remove_parents(state):
    new_state = deepcopy(state)
    new_state['incorrect'] = remove_duplicates(new_state['incorrect'])
    # remove parents
    new_state['incorrect'] = [guess for guess in new_state['incorrect'] if not any(are_parent_and_child(guess, other) for other in new_state['incorrect'] if guess != other)]
    changed = new_state != state
    return new_state, changed

def filter_incorrect_based_on_correct_guess(incorrect_guesses, correct_guess):
    guess_indices, guess_labels = correct_guess
    # Update incorrect guesses, remove those that are inconsistent with the correct guess
    new_incorrect_guesses = []
    for indices, g in incorrect_guesses:  # remove inconsistent guesses
        # if any entry is different than the one in the guess, remove it
        removed = False
        for ind, label in zip(indices, g):
            if (
                ind in guess_indices
                and label != guess_labels[guess_indices.index(ind)]
            ):
                removed = True
                break
        if not removed:  # filter
            new_indices = [i for i in indices if i not in guess_indices]
            new_g = [g[i] for i in range(len(g)) if indices[i] not in guess_indices]
            new_incorrect_guesses.append((new_indices, new_g))
    changed = new_incorrect_guesses != incorrect_guesses
    return new_incorrect_guesses, changed


def filter_based_on_labels(state):
    new_state = deepcopy(state)
    new_incorrect_guesses = new_state['incorrect']

    changed = False
    for label in new_state['labeled']:
        new_incorrect_guesses, has_changed = filter_incorrect_based_on_correct_guess(new_incorrect_guesses, ([label[0]], [label[1]]))
        changed = changed or has_changed

    new_state['incorrect'] = new_incorrect_guesses
    return new_state, changed

def create_new_labels(state):
    """Creates new labels based on those incorrect guesses of only one index"""
    new_state = deepcopy(state)
    new_incorrect_guesses = new_state['incorrect']
    new_labeled = new_state['labeled']

    changed = False
    ind_to_remove = []
    for i, guess in enumerate(new_incorrect_guesses):
        indices, labels = guess
        if len(indices) == 1:
            new_label = (indices[0], 1-labels[0])
            assert not any(
                new_label[0] == idx for idx, _ in new_labeled
            ), "annotation already present"
            new_labeled.append(new_label)
            changed = True
    new_incorrect_guesses = [inc_guess for ind, inc_guess in enumerate(new_incorrect_guesses) if ind not in ind_to_remove]
    return new_state, changed

def simplify_1bit_differences_individual(guess, other_guesses, allow_equal=False):
    """Remove one bit differences between guess and other_guesses."""
    to_remove = []
    for j, other_guess in enumerate(other_guesses):
        if not allow_equal and guess == other_guess:
            raise ValueError("guess shouldn't be equal to any of other_guesses")
        if guess != other_guess:
            if set(guess[0]) == set(other_guess[0]):  # the support is the same
                num_diffs = 0
                for ind, label in zip(guess[0], guess[1]):
                    if label != other_guess[1][other_guess[0].index(ind)]:
                        num_diffs += 1
                        if num_diffs == 1:
                            diff_at = ind
                        elif num_diffs > 1:
                            break
                if num_diffs == 1:
                    to_remove.append((j, diff_at))
    return to_remove

def remove_duplicates(incorrect):
    return [guess for i, guess in enumerate(incorrect) if guess not in incorrect[:i]]

def simplify_1bit_differences(state):
    """Remove one bit differences between incorrect guesses."""
    new_state = deepcopy(state)
    incorrect = remove_duplicates(new_state['incorrect'])

    i = 0
    while i < len(incorrect)-1:
        guess = incorrect[i]
        to_remove = simplify_1bit_differences_individual(guess, incorrect[i+1:], allow_equal=False)
        for j, diff_at in to_remove:
            # remove from j 
            indices_j, labels_j = incorrect[i+1:][j]
            ind = indices_j.index(diff_at)
            labels_j.pop(ind)
            indices_j.pop(ind)
        
        incorrect = remove_duplicates(incorrect)  # will keep the first one
        if len(to_remove) > 0:
            incorrect.pop(i)
            # i = 0  # start over
        else:
            i += 1

        

    changed = incorrect != state['incorrect']
    new_state['incorrect'] = incorrect
    return new_state, changed
    







def simplify_state(state, remove_parents_flag=True, filter_based_on_labels_flag=True, create_new_labels_flag=True, simplify_1bit_differences_flag=True):
    """Creates a simpler state without losing any information.
    We remove parents of incorrect guesses, filter incorrect guesses based on labels, simplify incorrect guesses if there are one bit differences and make incorrect guesses of length 1 new labels."""
    assert len(state['incorrect']) == 0 or len(state['incorrect'][0]) > 0, "incorrect guesses should be non-empty"
    new_state = deepcopy(state)

    while True:
        # remove_parents, i.e. those guesses that contain the current guess
        if remove_parents_flag:
            new_state, changed = remove_parents(new_state)
            remove_parents_flag = False
        
        if filter_based_on_labels_flag:
            new_state, changed = filter_based_on_labels(new_state)
            filter_based_on_labels_flag = False
            if changed:
                remove_parents_flag = True
                create_new_labels_flag = True
                simplify_1bit_differences_flag = True
                continue
        
        if create_new_labels_flag:
            new_state, changed = create_new_labels(new_state)
            create_new_labels_flag = False
            if changed:
                filter_based_on_labels_flag = True
                continue
        
        if simplify_1bit_differences_flag:
            new_state, changed = simplify_1bit_differences(new_state)
            simplify_1bit_differences_flag = False
            if changed:
                remove_parents_flag= True
                create_new_labels_flag = True
                simplify_1bit_differences_flag = True
                continue
        
        break

    return new_state
        




def update_state(state, guess, is_correct, efficient=True):
    """Updates the state by following some rules.
    1. If the guess is correct, we add those labels to the set of labeled examples.
    2. If the guess is correct, we remove the incorrect guesses that are inconsistent with it and remove the columns (indices) corresponding to the guess for all other incorrect guesses.
    3. If there is an incorrect guess with lenght=1, we annotate that label and create a new correct guess by flipping the original incorrect one (and we call this function again for this guess).
    4. If the guess is incorrect and matches up to 1 with another guess then remove the different bit from the already present guess. Do this for every already present incorrect guess.
    5. If the guess is incorrect and it's included in another incorrect guess present in the list then remove the parent.
    """
    if not efficient:
        if is_correct:
            for i, label in zip(guess[0], guess[1]):
                if (i, label) not in state['labeled']:
                    state['labeled'].append((i, label))
                else:
                    raise ValueError("label already present")
            return simplify_state(state)
        else:
            state['incorrect'].append(guess)
            return simplify_state(state)
    # bstate =    {'labeled': [], 'incorrect': [([0, 2, 3], [0, 1, 1]), ([1, 2, 3], [0, 0, 1]), ([0, 1, 3], [0, 0, 1]), ([0, 1, 2, 3], [1, 0, 1, 0]), ([0, 1, 2], [0, 1, 0])]}
    # bguess =    ([2], [0])
    # biscorrect = True
    # if state == bstate and guess == bguess and is_correct == biscorrect:
    #     breakpoint()
    if state != simplify_state(state):
        raise ValueError("state is not simplified")
    new_state = deepcopy(state)
    labeled, incorrect = new_state['labeled'], new_state['incorrect']
    guess_indices, guess_labels = guess



    # print("guess: ", guess)
    # print("labeled: ", labeled)
    # print("incorrect_guesses: ", incorrect_guesses)
    # breakpoint()

    if is_correct:
        # the intersection between current labeled indices and correct guess indices should be empty
        assert len(set(guess_indices).intersection(set([idx for idx, _ in labeled]))) == 0, "you should not guess things you know" 
        # Update incorrect guesses, remove those that are inconsistent with the correct guess
        incorrect, has_changed = filter_incorrect_based_on_correct_guess(incorrect, guess)
        # note that now some incorrect guesses could be used as labels, update the labels
        new_state = {"labeled": labeled, 'incorrect': incorrect}
        new_state = simplify_state(new_state, remove_parents_flag=True, filter_based_on_labels_flag=False, create_new_labels_flag=True, simplify_1bit_differences_flag=True)
        labeled, incorrect = new_state['labeled'], new_state['incorrect']

        # Update labeled examples if needed
        for i in guess_indices:  # add new labels
            if not any(i == idx for idx, _ in deepcopy(labeled)):
                labeled.append((i, guess_labels[guess_indices.index(i)]))
            else:
                print("label already present")

        new_state = {"labeled": labeled, 'incorrect': incorrect}
        new_state = simplify_state(new_state, remove_parents_flag=False, filter_based_on_labels_flag=True, create_new_labels_flag=False, simplify_1bit_differences_flag=False)
        
    else:
        if len(guess_indices) == 1:  # if guess is single then we have a correct guess
            # Annotate label
            new_correct_guess = ([guess_indices[0]], [1 - guess_labels[0]])
            assert not any(
                new_correct_guess[0] == idx for idx, _ in labeled
            ), "annotation already present"

            # Call this function again for this guess
            return update_state(
                {"labeled": labeled, 'incorrect': incorrect},
                new_correct_guess,
                True,
            )

        # Update incorrect guesses
        incorrect = [  # removes parents
            (indices, g)
            for indices, g in incorrect
            if not are_parent_and_child((indices, g), (guess_indices, guess_labels))
        ]

        # Remove the different bit from the already present guesses
        to_remove = simplify_1bit_differences_individual(guess, incorrect, allow_equal=False)
        create_new_labels_flag = False
        for j, diff_at in to_remove:
            ind = incorrect[j][0].index(diff_at)
            incorrect[j][0].pop(ind)
            incorrect[j][1].pop(ind)

            # new_state doesn't have the new guess yet. An old guess might now be a label
            if len(incorrect[j][0]) == 1:
                # we have a new correct guess
                create_new_labels_flag = True
            
        if len(to_remove) == 0:
            incorrect.append((guess_indices, guess_labels))
            new_state = {"labeled": labeled, 'incorrect': incorrect}
        else:
            # state shouldn't have changed here
            new_state = simplify_state(new_state, filter_based_on_labels_flag=False, simplify_1bit_differences_flag=True, create_new_labels_flag=create_new_labels_flag)
    try:
        simplified_state = simplify_state(new_state)
    except:
        breakpoint()
        simplified_state = simplify_state(new_state)
    try:
        if simplified_state != new_state:
            raise ValueError("simplify_state should not change the state")
    except:
        breakpoint()
    assert len(new_state['incorrect']) == 0 or len(new_state['incorrect'][0]) > 0, "incorrect guesses should be non-empty"

    return new_state

def generate_binary_vectors(N, n):
    results = []
    generate_binary_vectors_recursive(N, n, [], results)
    return sorted(results)

def generate_binary_vectors_recursive(N, n, current, results):
    if N == 0:
        if n == 0:
            results.append(current)
        return
    
    if n > 0:
        generate_binary_vectors_recursive(N-1, n-1, current + [1], results)
    
    generate_binary_vectors_recursive(N-1, n, current + [0], results)

def get_best_guess(guess_length, state, inputs, predictor):
    # computes the best guess given n. For n = 1, this should be an active learning method (to-do)
    # for now we will take the first unlabeled example
    labeled_indices = set([idx for idx, _ in state["labeled"]])
    unlabeled_indices = [i for i in range(len(inputs)) if not i in labeled_indices]
    if guess_length == 1:  # randomly sample one unlabeled index
        try:
            return ([np.random.choice(unlabeled_indices)], [0])
        except:
            breakpoint()
    else:
        # make a prediction over unlabeled inputs
        unlabeled_inputs = np.array([inputs[i] for i in unlabeled_indices])
        unlabeled_predictions = predictor(unlabeled_inputs)
        # get the guess_length most certain predictions
        certainty = np.abs(unlabeled_predictions - 0.5)
        ordered_certain_indices = (np.argsort(certainty)[::-1]).tolist()
        slack = 10

        sorted_certainties = certainty[ordered_certain_indices[:guess_length+slack]]
        sorted_certainties_list = sorted_certainties.tolist()
        def key_fn(x):
            # certainties = np.abs(np.array([sorted_certainties[i] for i in x['indices']]) - np.array(x['inv_mask']))
            certainties = np.array([sorted_certainties[i] for i in x['indices']])
            return np.sum(np.log(certainties))
        
        n_invertions = 0
        while n_invertions <= guess_length:
            invertion_masks = generate_binary_vectors(guess_length, n_invertions)
            subsets = [{'indices':[list(sorted_certainties).index(x) for x in c], 'inv_mask':inv_mask} for inv_mask in invertion_masks for c in itertools.combinations(sorted_certainties, guess_length)]
            # subsets.sort(key=lambda s: sum(log_certainties[i] for i in s), reverse=True)
            subsets.sort(key=key_fn, reverse=True)
            subsets = [{'indices':np.array(ordered_certain_indices)[s['indices']], 'inv_mask':s['inv_mask']} for s in subsets]
            # subsets = [{'indices':np.array(ordered_certain_indices)[s['indices']]} for s in subsets]

            for sset_dict in subsets:
                most_certain_indices = sset_dict['indices']
                predictions = np.around(unlabeled_predictions[most_certain_indices]).astype(int)
                predictions = np.abs(predictions - np.array(sset_dict['inv_mask']))
                guess = ([unlabeled_indices[i] for i in most_certain_indices], predictions.tolist())
                if are_consistent_inc_cor(state['incorrect'], guess):  # no incorrect guess is included in guess
                    return guess
            n_invertions += 1
        breakpoint()
        raise RuntimeError("No valid guess found")


###### TREE SEARCH ######
def prob_of_guess(guess, inputs, predictor):
    prob = 1
    for i, label in zip(guess[0], guess[1]):
        symbol_prob = predictor(np.array([inputs[i]]))
        prob *= symbol_prob**label * (1 - symbol_prob) ** (1 - label)
    return prob


def prob_intersection(guesses, inputs, predictor):
    # computes the probability of the intersection of guesses using conditional probabilities
    prob = 1
    prev_guess = ([], [])
    for k in range(len(guesses)):
        guess = guesses[k]
        if not are_consistent_cor_cor(prev_guess, guess):
            return 0
        guess = take_differing_bits(guess, prev_guess)
        prob *= prob_of_guess(guess, inputs, predictor)
        prev_guess = (prev_guess[0] + guess[0], prev_guess[1] + guess[1])
    return prob


def incorrect_prob(incorrect_guesses, inputs, predictor):
    # computes the probability of all the incorrect guesses being incorrect given by the predictor
    # we first compute the probability of at least one of them being correct
    prob_union = 0
    guesses = (
        incorrect_guesses  # change the name because we want at least one to be true
    )
    # for k in tqdm.tqdm(range(1, len(guesses) + 1)):  # takes time
    for k in range(1, len(guesses) + 1):
        for some_guesses in itertools.combinations(guesses, k):
            prob_inter = prob_intersection(some_guesses, inputs, predictor)
            # update union probability
            prob_union += (-1) ** (k + 1) * prob_inter
    return 1 - prob_union


def compute_prob(inputs, predictor, state, guess):
    # computes the probability of the guess being correct given by the predictor and the state
    guess_indices, guess_labels = guess
    labeled_indices = [i for i, _ in state["labeled"]]
    assert not any([i in labeled_indices for i in guess_indices])
    incorrect_guesses = state['incorrect']
    if not are_consistent_inc_cor(incorrect_guesses, guess):
        breakpoint()
        raise ValueError("Provided guess is inconsistent with current state")
    incorrect_guesses_given_guess = update_state(state, guess, True)[
        'incorrect'
    ]
    prob = 1
    prob *= incorrect_prob(incorrect_guesses_given_guess, inputs, predictor)
    prob *= prob_of_guess(guess, inputs, predictor)
    prob /= incorrect_prob(incorrect_guesses, inputs, predictor)
    return prob


# Define the transition function
def transition(guess, state, inputs, predictor):
    # Return a list of possible next states and their probabilities given the current state and action
    pos_state = update_state(state, guess, True)
    neg_state = update_state(state, guess, False)
    pos_prob = compute_prob(inputs, predictor, state, guess)
    neg_prob = 1 - pos_prob
    return [(pos_state, pos_prob), (neg_state, neg_prob)]


def count_possible_guesses(incorrect_guesses):
    support_incorrect = sorted(list(set([i for indices, _ in incorrect_guesses for i in indices])))  # support of incorrect guesses
    count = 0
    for v in range(2**len(support_incorrect)):
        makes_them_incorrect = True
        for inc_guess in incorrect_guesses:
            # check if this one is correct given the vector
            # first create the binary vector corresopnding to v
            all_digits_ok = True
            binary_v = [int(i) for i in list(bin(v)[2:])] 
            binary_v += [0] * (len(support_incorrect) - len(binary_v))
            for ind, index in enumerate(inc_guess[0]):
                guessed_label = inc_guess[1][ind]
                actual_label = binary_v[support_incorrect.index(index)]
                if guessed_label != actual_label:
                    all_digits_ok = False
                    break
            if all_digits_ok:
                makes_them_incorrect = False
                break
        if makes_them_incorrect:
            count += 1
    return count, len(support_incorrect)



# Define the reward function
def approx_cost_function(state, inputs, predictor):
    # Return the reward of a state when depth=0
    # this case will be the length of the set of possible labelings
    N = len(inputs)
    n = len(state["labeled"])
    # count the number of possible labelings
    possible, support_size = count_possible_guesses(state['incorrect'])
    log_number_of_possible_labelings = (N - n - support_size) + np.log2(1 + possible / (2**(N - n - support_size)))
    print(f"Number of possible guesses: {possible}")
    print(f"Cost: {log_number_of_possible_labelings}")
    return log_number_of_possible_labelings


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


def select_node(state_node, N):
    update_priorities(state_node)
    leaves = get_leaves(state_node)
    non_terminal_leaves = [leaf for leaf in leaves if len(leaf.state["labeled"]) < N]
    if len(non_terminal_leaves) == 0:
        return 'all nodes expanded'
    return leaves[np.argmax([leaf.priority for leaf in non_terminal_leaves])]


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
    action_node.cost = 0
    for child_ind, state_child in enumerate(action_node.state_children):
        action_node.cost += state_child.cost * action_node.children_probs[child_ind]


def expand_node(state_node, max_n, inputs, predictor):
    # expand a state node by considering actions in increasing n and their implied states
    best_cost = state_node.cost
    N = inputs.shape[0]
    n = len(state_node.action_children) + 1
    while n <= max_n and n <= (N - len(state_node.state['labeled'])):
        # expand
        best_guess = get_best_guess(n, state_node.state, inputs, predictor)
        next_states = transition(best_guess, state_node.state, inputs, predictor)
        action_child = ActionNode(best_guess, state_node)
        action_child.add_state_children(next_states)
        state_node.action_children.append(action_child)
        # udpate
        for state_child in action_child.state_children:
            state_child.cost = approx_cost_function(state_child.state, inputs, predictor)
        update_action_cost_depth_0(action_child)

        if action_child.cost < best_cost:
            best_cost = action_child.cost
        else:
            break
        n += 1
    state_node.cost = best_cost


def update_parents(state_node, max_n, inputs, predictor):
    # update parents of state node by going up, computing stop criterion and expanding if needed.
    # Continue upwards if chaning the value function of parent state node
    if state_node.parent is None:
        return
    parent_action = state_node.parent
    update_action_cost_depth_0(parent_action)
    parent_state = parent_action.parent
    if parent_action.cost < parent_state.cost:
        parent_state.cost = parent_action.cost
        if (
            parent_state.cost == parent_state.action_children[-1].cost
        ):  # if more actions are needed
            expand_node(parent_state, max_n, inputs, predictor)
        update_parents(
            parent_state, max_n, inputs, predictor
        )  # assume they're up to date unless parent_state.cost changes


def get_best_action(state_node):
    # get the best action from a state node
    return state_node.action_children[
        np.argmin([action_child.cost for action_child in state_node.action_children])
    ]

def get_node_with_state(root_node, state):
    """Explores the tree and returns the node with the given state using breadth first """
    queue = [root_node]
    while len(queue) > 0:
        node = queue.pop(0)
        if node.state == state:
            return node
        for action_child in node.action_children:
            queue.extend(action_child.state_children)
    return None

def initialize_tree(root_state):
    root_node = StateNode(root_state)
    root_node.priority = 1  # trickle down priorities
    return root_node

def tree_search(root_node, N, max_expansions, max_n, inputs, predictor):

    print("Expanding tree...")
    for i in tqdm.tqdm(range(max_expansions)):
        node = select_node(root_node, N)
        if node == 'all nodes expanded':
            break
        expand_node(node, max_n, inputs, predictor)
        update_parents(node, max_n, inputs, predictor)
    

    best_action = get_best_action(root_node)

    print("Tree search results:")
    print(root_node.state)
    print(best_action)
    return best_action.guess, best_action.cost
