"""This script simulates some simple annotation problems. 
These problems will be solved exactly using Huffman encoding and exhaustive tree search.
Then we will truncate the tree and solve the problem approximately using our methods.
Lastly we will add noise to the predictor.
"""
import os
import shutil
from pathlib import Path
import numpy as np
import sklearn
import sklearn.datasets
import tqdm
import json

import huffman
import simple_tree_search
from simple_tree_search import STS, entropy_given_probs_binary, entropy_given_state_preds_binary
import visualization as vis

#### data ####
def generate_2d_gaussians_and_predictor(N, pos_center=[2]*2, pos_ratio=0.5):
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

def generate_2d_gaussians_with_linear_and_predictor(N, pos_center=[3]*2, temperature=0.3, std=0.5):
    # get samples
    assert N % 2 == 0
    cov = [[std, 0], [0, std]]
    samples1 = np.random.multivariate_normal(
        [0, 0], cov, N // 2
    )
    samples2 = np.random.multivariate_normal(
        pos_center, cov, N // 2
    )
    # create a linear predictor whose decision boundary is the line y = x
    def prob_of_being_positive(x):
        # compute the distance to the line y = x
        dist = (x[:, 1] - x[:, 0]) / np.sqrt(2)
        return 1 / (1 + np.exp(-dist / temperature))

    samples = np.concatenate([samples1, samples2])
    probs = prob_of_being_positive(samples)
    labels = np.random.rand(N) < probs
    pos_samples = samples[labels]
    neg_samples = samples[~labels]

    return pos_samples, neg_samples, prob_of_being_positive

def generate_2d_classification_and_predictor(N):
    X, y = sklearn.datasets.make_classification(n_samples=N, n_features=2, n_classes=2, n_informative=1, n_redundant=0, n_repeated=0, n_clusters_per_class=1, flip_y=0.2)
    pos_samples = X[y==1]
    neg_samples = X[y==0]
    cls = sklearn.linear_model.LogisticRegression()
    cls.fit(X, y)
    def prob_of_being_positive(x):
        return cls.predict_proba(x)[:, 1]
    return pos_samples, neg_samples, prob_of_being_positive




def extend_results(results, name, value):
    if name not in results:
        results[name] = [value]
    else:
        results[name].append(value)
    return results


########### functions to run methods ############
def run_STS(predictions, labels, results, cost_fn="entropy"):
    N = len(labels)
    # simple tree search
    alia = STS(
        max_expansions=8, al_method="uncertainty", max_n=8, cost_fn=cost_fn, reduce_certainty_factor=0.01, reset_tree=False
    )
    state = {"indices": np.arange(N), "incorrect": None}
    root_node = STS.initialize_root_node(state)

    ### query ###
    probs_and_outcomes = []
    estimated_costs = []
    n_questions = 0
    while len(root_node.state["indices"]) > 0:  # still things to annotate
        n_questions += 1
        new_predictions = predictions[state["indices"]]
        alia.set_unlabeled_predictions(
            (state["indices"], new_predictions)
        )
        best_question_node = alia.tree_search(root_node)
        question, optimal_cost, correct_prob = best_question_node.guess, best_question_node.cost, best_question_node.children_probs[1]
        entropy1 = entropy_given_state_preds_binary(root_node.state, (state["indices"], new_predictions))
        entropy2 = entropy_given_probs_binary(new_predictions)
        # if optimal_cost > entropy1 + 1:
        #     condition = n_questions == 1 or ((root_node.state['incorrect'] is not None) and (prev == 'above'))
        #     print("above", condition)
        #     prev = 'above'
        # else:
        #     condition = n_questions == 1 or ((root_node.state['incorrect'] is not None) and (prev == 'above'))
        #     print("below", condition)
        #     prev = 'below'
        estimated_costs.append((optimal_cost, entropy1, entropy2, n_questions))
        assert 0 < len(question[0]), "you should ask something"
        answer = bool((question[1] == labels[question[0]]).all())
        probs_and_outcomes.append((correct_prob, answer))
        state = STS.update_state(root_node.state, question, answer)
        root_node = STS.set_new_root_node(best_question_node.state_children[answer*1], destroy_descendants=False)
    extend_results(results, "probs_and_outcomes_sts", probs_and_outcomes)
    extend_results(results, "estimated_costs_sts", estimated_costs)
    extend_results(results, "n_questions_sts", n_questions)
    # print('-'*20)
    return results


def run_huffman(predictions, labels, results):
    label_as_index = huffman.get_power_01(N).index(labels.tolist())
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
    return results




####### main #######


def main(n_seeds, N, generator_fn, dstdir):
    reset = False
    Path(dstdir).mkdir(parents=True, exist_ok=True)
    dstfilename = f"{n_seeds}_{N}_{generator_fn.__name__}"
    print(f"Running {dstfilename}")
    dstdir = Path(dstdir)
    (dstdir / "tmp").mkdir(parents=True, exist_ok=True)
    shutil.rmtree(dstdir / "tmp")
    records_filename = dstdir / f"{dstfilename}.json"
    # custom
    results = {}
    if not os.path.exists(records_filename) or reset:
        for seed in tqdm.tqdm(range(n_seeds)):
        # for seed in range(n_seeds):
            np.random.seed(seed)
            (
                pos_samples,
                neg_samples,
                prob_of_being_positive,
            ) = generator_fn(N)
            predictions = prob_of_being_positive(
                np.concatenate((neg_samples, pos_samples), axis=0)
            )
            labels = np.array([0] * len(neg_samples) + [1] * len(pos_samples))
            entropy = simple_tree_search.entropy_given_probs_binary(predictions)
            extend_results(results, "entropies", entropy)

            results = run_STS(predictions, labels, results)
            results = run_huffman(predictions, labels, results)

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
    vis.scatter_probs_and_outcomes(
        data["probs_and_outcomes_sts"],
        dstdir / f"{dstfilename}_scatter_probs_and_outcomes_sts",
    )
    vis.visualize_sts_estimated_cost(data['estimated_costs_sts'], dstdir / f"{dstfilename}_sts_estimated_cost")


if __name__ == "__main__":
    n_seeds, N, dstdir = 1000, 10, "results/simulations"
    Path(dstdir).mkdir(parents=True, exist_ok=True)
    dstfilename = f"{n_seeds}_{N}"
    # import cProfile
    # command = (
    #     f"main({n_seeds}, {N}, {pos_center}, {pos_ratio}, '{dstdir}', '{dstfilename}')"
    # )
    # cProfile.run(
    #     command, sort="cumtime", filename=f"{dstdir}/{dstfilename}_simulation.profile"
    # )
    np.random.seed(42)
    for generator_fn in [
        generate_2d_gaussians_and_predictor,
        generate_2d_gaussians_with_linear_and_predictor,
        generate_2d_classification_and_predictor]:
        # (
        #     pos_samples,
        #     neg_samples,
        #     prob_of_being_positive,
        # ) = generator_fn(N)
        # X = np.concatenate((neg_samples, pos_samples), axis=0)
        # y = np.array([0] * len(neg_samples) + [1] * len(pos_samples))
        # X_to_annotate = X
        # y_to_annotate = y
        # indices_to_ask = []
        # predictions = []
        # X_annotated = np.empty((0, 2))
        # y_annotated = np.empty((0, 1))

        # vis.plot_model_outputs(
        #     prob_of_being_positive,
        #     None,
        #     X,
        #     y,
        #     X_to_annotate,
        #     y_to_annotate,
        #     indices_to_ask,
        #     predictions,
        #     X_annotated,
        #     y_annotated,
        #     generator_fn.__name__,
        # )


        main(n_seeds, N, generator_fn, dstdir)

