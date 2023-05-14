import json
from copy import deepcopy
import os
import matplotlib.pyplot as plt
import time
import tqdm
import numpy as np

"""This is a program that checks if an update function for the state is correct.
We do some guesses of different lengths and we want to reduce redundancy in the state.
A state with no redundancy is the one with maximum number of labeled indices and minimum number of incorrect guesses (and of minimum length), without losing any information.
"""
from tree_and_state import get_power_01, are_parent_and_child, update_state


def create_new_guess(avail_indices, incorrect_guesses):
    """Generates a new guess that isn't in the list of incorrect guesses and isn't a parent of any of the incorrect guesses.
    `avail_indices` is a list of indices that are available for guessing."""
    st = time.time()
    while True:  # create a new guess
        # now we make random guesses. We first sample a guess length
        guess_length = np.random.randint(1, len(avail_indices) + 1)
        # now we choose guess_length indices to guess
        guess_indices = sorted(
            list(np.random.choice(avail_indices, guess_length, replace=False))
        )
        # now we create a random binary guess over those indices
        guess_labels = list(np.random.randint(2, size=guess_length))
        guess = (guess_indices, guess_labels)
        is_valid = (guess not in incorrect_guesses) and not any(
            [
                are_parent_and_child(guess, incorrect_guess)
                for incorrect_guess in incorrect_guesses
            ]
        )
        if is_valid:
            break
        if time.time() - st > 5:
            breakpoint()
    return guess


def state_is_correct(state, L):
    """Checks that state corresponds to L.
    L is a list of binary lists or vectors.
    The check involves the computation of the possible labelings from the state."""
    # print("Possible labelings ", L)
    # print("State ", state)
    # breakpoint()
    state_possible_labelings = get_power_01(len(L[0]))
    for i, label in state["labeled"]:
        values_to_remove = []
        for possible_labeling in state_possible_labelings:
            if possible_labeling[i] != label:
                values_to_remove.append(possible_labeling)
        for v in values_to_remove:
            state_possible_labelings.remove(v)
    for indices, guess in state["incorrect"]:
        values_to_remove = []
        for possible_labeling in state_possible_labelings:
            if np.all(np.array(possible_labeling)[indices] == np.array(guess)):
                values_to_remove.append(possible_labeling)
        for v in values_to_remove:
            state_possible_labelings.remove(v)
    # remove indices_to_remove from state_possible_labelings

    is_correct = state_possible_labelings == L
    if not is_correct:
        print("state:", state)
        print("state_possible_labelings:", state_possible_labelings)
        print("L:", L)
        raise ValueError("state is not correct")
    return is_correct


def update_L(L, new_guess, is_correct):
    guess_indices, guess_labels = new_guess
    L = [
        l
        for l in L
        if np.all(np.array(l)[guess_indices] == np.array(guess_labels)) == is_correct
    ]
    return L


def get_times(seed, size, use_L, check_correctness):
    assert size >= 1
    np.random.seed(seed)
    # we play with vectors of size 5 of which we have 32
    # all possible binary vectors of size 5
    power_01 = get_power_01(size)
    labeled = []  # list of (index, label) tuples
    incorrect_guesses = []  # list of (indices, guess) tuples
    state = {"labeled": labeled, "incorrect": incorrect_guesses}
    # get a random binary vector of size 5
    if use_L:
        L = deepcopy(power_01)  # possible labelings
    y = list(np.random.randint(2, size=size))
    update_times = {"L": 0, "state": 0} if use_L else {"state": 0}
    n_updates = 0
    # print("<<<<____________________________________>>>>")
    while True:
        # print("=====================================")
        # print("Possible labelings: ", L)
        # print("Number of possible labelings: ", len(L))
        # print("Current state: ", state)
        # print("Correct labeling is: ", y)
        avail_indices = [
            i for i in range(size) if i not in [ind for ind, lab in state["labeled"]]
        ]
        new_guess = create_new_guess(avail_indices, state["incorrect"])
        # print("New guess: ", new_guess)
        guess_indices, guess_labels = new_guess
        # now we check if the guess is correct
        is_correct = np.all(
            np.array([y[i] for i in guess_indices]) == np.array(guess_labels)
        )
        # print("Correct guess!" if is_correct else "Incorrect guess")
        # now we update the set of possible labelings
        if use_L:
            st = time.time()
            L = update_L(L, new_guess, is_correct)
            update_times["L"] = update_times["L"] * n_updates / (n_updates + 1) + (
                time.time() - st
            ) / (n_updates + 1)
        # now we update our state
        st = time.time()
        try:
            state = update_state(state, (guess_indices, guess_labels), is_correct)
        except AssertionError as e:
            print("=====================================")
            print("Correct labeling is: ", y)
            print("Current state: ", state)
            print("New guess: ", new_guess)
            print("Correct guess! " if is_correct else "Incorrect guess")
            print("Expected L", L)
            breakpoint()
            state = update_state(state, (guess_indices, guess_labels), is_correct)

        update_times["state"] = update_times["state"] * n_updates / (n_updates + 1) + (
            time.time() - st
        ) / (n_updates + 1)
        if use_L:
            assert state_is_correct(state, L)
        n_updates += 1
        if len(state["labeled"]) == size:
            break
    # print("=====================================")
    # print("Completed successfully!")
    # print("Last state: ", state)
    # print(">>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<")
    return update_times


if __name__ == "__main__":
    n_seeds = 10000
    max_size = 6
    use_L = True
    check_correctness = True
    dst_filename = f"results_{n_seeds}_{max_size}_{use_L}_{np.random.rand()}.json"
    if not os.path.exists(dst_filename):
        if use_L:
            Ltimes = {}
        statetimes = {}
        for size in tqdm.tqdm(range(1, max_size), desc="size"):
            if use_L:
                Ltimes[size] = 0
            statetimes[size] = 0
            for seed in tqdm.tqdm(range(n_seeds), desc="seed", leave=False):
                try:
                    # size, seed = 4, 902
                    utimes = get_times(seed, size, use_L, check_correctness)
                except Exception as e:
                    print("size:", size)
                    print("seed:", seed)
                    raise e

                if use_L:
                    Ltimes[size] += utimes["L"] / n_seeds
                statetimes[size] += utimes["state"] / n_seeds
            with open(dst_filename, "w") as f:
                if use_L:
                    json.dump({"L": Ltimes, "state": statetimes}, f)
                else:
                    json.dump({"state": statetimes}, f)
    else:
        print("File already exists, loading it")

    with open(dst_filename, "r") as f:
        data = json.load(f)
    if use_L:
        Ltimes = data["L"]
    statetimes = data["state"]

    # plt.figure()
    # if use_L:
    #     plt.plot(list(Ltimes.keys()), list(Ltimes.values()), label='L')
    # plt.plot(list(statetimes.keys()), list(statetimes.values()), label='labels + compressed incorrect guesses')
    # plt.ylabel('time ($s$)')
    # plt.xlabel('dataset size $N$')
    # plt.legend()
    # plt.savefig(dst_filename.replace('.json', '.png'))

    # plt.figure()
    # if use_L:
    #     plt.plot(list(Ltimes.keys()), np.log(list(Ltimes.values())), label='L')
    # plt.plot(list(statetimes.keys()), np.log(list(statetimes.values())), label='labels + compressed incorrect guesses')
    # plt.ylabel('log time ($log_e s$)')
    # plt.xlabel('dataset size $N$')
    # plt.legend()
    # plt.savefig('log_'+dst_filename.replace('.json', '.png'))
