"""This script simulates some simple annotation problems. 
These problems will be solved exactly using Huffman encoding and exhaustive tree search.
Then we will truncate the tree and solve the problem approximately using our methods.
Lastly we will add noise to the predictor.
"""
import os
import shutil
from pathlib import Path
import numpy as np
import tqdm
import json

import general_binary_tree_search as gbts
import huffman
import simple_tree_search
import visualization as vis


#### data ####
def generate_2d_gaussians_and_predictor(N, pos_center, pos_ratio):
    # get samples
    pos_ratio = int(N * pos_ratio) / N  # make exact
    neg_samples = np.random.multivariate_normal(
        [0, 0], [[1, 0], [0, 1]], int(N * (1 - pos_ratio))
    )
    pos_center = [2, 2]
    pos_samples = np.random.multivariate_normal(
        pos_center, [[1, 0], [0, 1]], int(N * pos_ratio)
    )
    assert len(pos_samples) + len(neg_samples) == N

    # our predictor is given by the same gaussians that generate the data
    def prob_neg_fn(x):
        # the probability of belonging to the negative class is given by the probability of the negative 2d gaussian
        return 1 / (2 * np.pi) * np.exp(-0.5 * (x[:, 0] ** 2 + x[:, 1] ** 2))

    def prob_pos_fn(x):
        # the probability of belonging to the positive class is given by the probability of the positive 2d gaussian
        return (
            1
            / (2 * np.pi)
            * np.exp(
                -0.5 * ((x[:, 0] - pos_center[0]) ** 2 + (x[:, 1] - pos_center[1]) ** 2)
            )
        )

    def prob_of_being_positive(x):
        # P(y=1|x) = P(x|y=1)P(y=1) / (P(x, y=1) + P(x, y=0)) = P(x|y=1)P(y=1) / (P(x|y=1)P(y=1) + P(x|y=0)P(y=0))
        px_given_y1_times_py1 = prob_pos_fn(x) * pos_ratio
        py1_given_x = px_given_y1_times_py1 / (
            px_given_y1_times_py1 + prob_neg_fn(x) * (1 - pos_ratio)
        )
        return py1_given_x

    return pos_samples, neg_samples, prob_of_being_positive


def extend_results(results, name, value):
    if name not in results:
        results[name] = [value]
    else:
        results[name].append(value)
    return results


####### main #######


def main(n_seeds, N, pos_center, pos_ratio, dstdir, dstfilename):
    reset = False
    Path(dstdir).mkdir(parents=True, exist_ok=True)
    dstfilename = f"{n_seeds}_{N}_{pos_center}_{pos_ratio}"
    print(f"Running {dstfilename}")
    dstdir = Path(dstdir)
    (dstdir / "tmp").mkdir(parents=True, exist_ok=True)
    shutil.rmtree(dstdir / "tmp")
    records_filename = dstdir / f"{dstfilename}.json"
    pos_center = [pos_center] * 2
    # custom
    results = {}
    # # #
    if not os.path.exists(records_filename) or reset:
        for seed in tqdm.tqdm(range(n_seeds)):
            np.random.seed(seed)
            (
                pos_samples,
                neg_samples,
                prob_of_being_positive,
            ) = generate_2d_gaussians_and_predictor(N, pos_center, pos_ratio)
            predictions = prob_of_being_positive(
                np.concatenate((neg_samples, pos_samples), axis=0)
            )
            labels = np.array([0] * len(neg_samples) + [1] * len(pos_samples))
            entropy = simple_tree_search.entropy_given_probs_binary(predictions)
            extend_results(results, "entropies", entropy)
            label_as_index = huffman.get_power_01(N).index(labels.tolist())

            # simple tree search
            sts = simple_tree_search.STS(
                max_expansions=100, al_method=None, max_n=10, cost_fn="entropy"
            )
            initial_state = {"indices": list(range(N)), "incorrect": None}
            root_node = sts.get_root_node(initial_state)
            new_predictions = (list(range(len(predictions))), list(predictions))
            sts.set_unlabeled_predictions(new_predictions)
            probs_and_outcomes = []
            estimated_costs = []
            labels_so_far = ([], [])
            n_questions = 0
            while len(root_node.state["indices"]) > 0:  # still things to annotate
                n_questions += 1
                question, optimal_cost, correct_prob = sts.tree_search(root_node)
                estimated_costs.append(optimal_cost)
                assert 0 < len(question[0]), "you should ask something"
                answer = all(
                    [
                        prediction == labels[index]
                        for index, prediction in zip(*question)
                    ]
                )
                probs_and_outcomes.append((correct_prob, answer))
                raise NotImplementedError("all the update is wrong because of the set destroying orders")
                # if answer or len(question[0]) == 1:
                #     labels_so_far[0].extend(question[0])
                #     labels_so_far[1].extend(question[1] if answer else (1-np.array(question[1])).tolist())
                #     new_state = {
                #         "indices": list(set(range(N)) - set(labels_so_far[0])),
                #         "incorrect": None,
                #     }
                # else:
                #     new_state = {
                #         "indices": root_node.state["indices"],
                #         "incorrect": question,
                #     }
                new_predictions = predictions[new_state["indices"]]
                sts.set_unlabeled_predictions((new_state['indices'], new_predictions.tolist()))
                root_node = sts.get_root_node(new_state)
            extend_results(results, "probs_and_outcomes_sts", probs_and_outcomes)
            extend_results(results, "estimated_costs_sts", estimated_costs)
            extend_results(results, "n_questions_sts", n_questions)

            # huffman
            huffman_tree, preds_for_vectors = huffman.huffman_encoding(predictions)
            analysis_result = huffman.analyse_huffman_tree(
                huffman_tree, preds_for_vectors
            )
            extend_results(results, "huffman_analysis", analysis_result)
            # vis.visualize_huffman_tree(huffman_tree, 'huffman_tree', preds_for_vectors)

            # run huffman questioning
            assert label_as_index in huffman_tree.indices
            node = huffman_tree
            n_questions = 0
            while node.indices != [label_as_index]:
                n_questions += 1  # ask one question and move down the latter
                response = label_as_index in node.children[0].indices
                if response:
                    node = node.children[0]
                else:
                    node = node.children[1]
            extend_results(results, "n_questions_huffman", n_questions)

            # # general binary tree search
            # max_expansions = 100
            # n_questions = 0
            # initial_state = list(range(2**N))  # state are indices
            # root_node = gbts.initialize_tree(initial_state)
            # probs_and_outcomes = []
            # estimated_costs = []
            # while len(root_node.indices) > 1:
            #     n_questions += 1
            #     question, optimal_cost, correct_prob = gbts.tree_search(
            #         root_node, max_expansions, preds_for_vectors
            #     )
            #     estimated_costs.append(optimal_cost)
            #     answer = label_as_index in question
            #     probs_and_outcomes.append((correct_prob, answer))
            #     new_indices = (
            #         question if answer else list(set(root_node.indices) - set(question))
            #     )
            #     root_node = gbts.get_node_with_indices(root_node, new_indices)
            #     root_node.priority = 1
            #     root_node.parent = None
            # extend_results(results, "probs_and_outcomes_gbts", probs_and_outcomes)
            # extend_results(results, "estimated_costs_gbts", estimated_costs)
            # extend_results(results, "n_questions_gbts", n_questions)

        with open(records_filename, "w") as f:
            json.dump(results, f)
    with open(records_filename, "r") as f:
        data = json.load(f)
        analysis = data["huffman_analysis"]
    vis.visualize_huffman_analysis(analysis, dstdir / f"{dstfilename}_huffman_analysis")
    vis.box_and_whisker_plot(
        data,
        dstdir / f"{dstfilename}_box_and_whisker",
    )
    vis.scatter_probs_and_outcomes(data['probs_and_outcomes_sts'], dstdir / f"{dstfilename}_scatter_probs_and_outcomes_sts")


if __name__ == "__main__":
    n_seeds, N, pos_center, pos_ratio, dstdir = 100, 8, 1, 0.5, "simulations"
    Path(dstdir).mkdir(parents=True, exist_ok=True)
    dstfilename = f"{n_seeds}_{N}_{pos_center}_{pos_ratio}"
    # import cProfile
    # command = (
    #     f"main({n_seeds}, {N}, {pos_center}, {pos_ratio}, '{dstdir}', '{dstfilename}')"
    # )
    # cProfile.run(
    #     command, sort="cumtime", filename=f"{dstdir}/{dstfilename}_simulation.profile"
    # )
    main(n_seeds, N, pos_center, pos_ratio, dstdir, dstfilename)
