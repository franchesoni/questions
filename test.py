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

def test_generate_binary_vectors():
    # Test Case 1: N=3, n=2
    expected_result = sorted([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
    result = generate_binary_vectors(3, 2)
    assert result == expected_result, f"Test Case 1 failed. Expected: {expected_result}, Got: {result}"

    # Test Case 2: N=4, n=1
    expected_result = sorted([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    result = generate_binary_vectors(4, 1)
    assert result == expected_result, f"Test Case 2 failed. Expected: {expected_result}, Got: {result}"

    # Test Case 3: N=5, n=3
    expected_result = sorted([[1, 1, 1, 0, 0], [1, 1, 0, 1, 0], [1, 1, 0, 0, 1], [1, 0, 1, 1, 0], [1, 0, 1, 0, 1],
                       [1, 0, 0, 1, 1], [0, 1, 1, 1, 0], [0, 1, 1, 0, 1], [0, 1, 0, 1, 1], [0, 0, 1, 1, 1]])
    result = generate_binary_vectors(5, 3)
    assert result == expected_result, f"Test Case 3 failed. Expected: {expected_result}, Got: {result}"

    print("All test cases passed!")

# Run the test
test_generate_binary_vectors()
print(generate_binary_vectors(5, 0))

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
        certainty = np.maximum(unlabeled_predictions, 1 - unlabeled_predictions)
        ordered_certain_indices = (np.argsort(certainty)[::-1]).tolist()
        slack = 10

        sorted_certainties = certainty[ordered_certain_indices[:guess_length+slack]]  # original indices are in ordered_certain_indices
        sorted_certainties_list = sorted_certainties.tolist()
        def key_fn(x):
            certainties = np.abs(np.array([sorted_certainties[i] for i in x['indices']]) - np.array(x['inv_mask']))
            return np.sum(np.log(certainties))
        n_invertions = 0
        while True: 
            # get all possible invertion masks of the subsets given n_invertions
            # these are all binary vectors of size guess_length with n_invertions ones
            invertion_masks = generate_binary_vectors(guess_length, n_invertions)
            # get all possible subsets of size guess_length masked by all possible invertion_masks
            subsets = [{'indices':[sorted_certainties_list.index(x) for x in c], 'inv_mask':inv_mask} for inv_mask in invertion_masks for c in itertools.combinations(sorted_certainties, guess_length)]
            subsets.sort(key=key_fn, reverse=True)
            # subsets = [np.array(ordered_certain_indices)[s['indices']] for s in subsets]

            for subset_dict in subsets:
                most_certain_indices = np.array(subset_dict['indices'])
                predictions = np.around(unlabeled_predictions[most_certain_indices]).astype(int)
                predictions = np.abs(predictions - np.array(subset_dict['inv_mask']))
                # invert n_invertions predictions using 1-pred starting from the least certains (the last)
                guess = ([unlabeled_indices[i] for i in most_certain_indices], predictions.tolist())
                if are_consistent_inc_cor(state['incorrect'], guess):  # no incorrect guess is included in guess
                    return guess
            n_invertions += 1
            breakpoint()
        raise RuntimeError("No valid guess found")


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

        log_certainties = np.log(certainty)[ordered_certain_indices[:guess_length+slack]]  # original indices are in ordered_certain_indices
        subsets = [[list(log_certainties).index(x) for x in c] for c in itertools.combinations(log_certainties, guess_length)]
        subsets.sort(key=lambda s: sum(log_certainties[i] for i in s), reverse=True)
        subsets = [np.array(ordered_certain_indices)[s] for s in subsets]

        for most_certain_indices in subsets:

        # for i in range(len(unlabeled_predictions) - guess_length + 1):
        #     most_certain_indices = ordered_certain_indices[: guess_length - 1] + [
        #         ordered_certain_indices[guess_length - 1 + i]
        #     ]

            predictions = np.around(unlabeled_predictions[most_certain_indices]).astype(int)
            guess = ([unlabeled_indices[i] for i in most_certain_indices], predictions.tolist())
            if are_consistent_inc_cor(state['incorrect'], guess):  # no incorrect guess is included in guess
                return guess
        breakpoint()
        raise RuntimeError("No valid guess found")

