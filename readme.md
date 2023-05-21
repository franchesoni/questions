# This is the repository for the paper
anony

## Experiments
Experiments are run from `main.py`. These are:
1. Run Huffman and STS for N=10
   - in these problems:
        - balanced
            - blobs
            - blobs + linear
            - make classification
            - spiral
        - unbalanced
            - make classification
    - report (for many seeds)
      - entropy (general, huffman)
      - expected number of questions (huffman)
      - cost function (sts)
      - number of questions
      - number of questions - entropy
2. Run STS over real datasets using an oracle
    - train an oracle
    - geq5, last
    - report
        - number of labels vs number of questions
        - prob correct vs outcomes
        - cost vs entropy
        - try buffering
3. Run STS over real datasets from scratch


# to-do
- use np arrays instead of lists for predictions
- check why points lie "above"
- test sts
    - probs
    - entropy
    - gaussian visualization
    - (optional) decision tree
    - mnist geq5 histogram visualization
- launch exp
- (optional) use huffman for incorrect resolution
- make presentation
- build demo

    - check mnist questions
    - run alia the gaussians
    - run alia with a bigger network on spirals
    - mnist geq5 histogram visualization
- make n=1 guesses consistent with prediction
- print prediction prob next to question

- problems with graph
    - duplicate nodes
    - prob of outcome for n=1 should be 1


- implement my method
- run my method and check number of questions

- - make faster
-     - jean zay A100
-     - torch 2.0 compile
-     - inmemory + indevice crop and normalize (horizontal flip) once per ds
-     - repeat ds + increase batch size
-     - amp

- implement entropy or margin sampling

- log intermediate steps
    - original cost and new best cost to understand the bias in the cost function

- mention that optimization adavances with computation but data doesn't 

- improve first is shortened second check by considering the labels

other optimizations:
    - once we made a wrong guess, solve the annotation of those points using huffman with the updated probabilities
    - use exponential search + binary search for the best n
    - make the model less confident 
    - train the model with a calibration loss

look more at:
- oracles that provide counterexamples
- queries of the type "these two are different or similar?"
    - this is querying q = (y_i, y_j) \in [(0, 1), (1, 0)] ?


Active Learning Using Arbitrary Binary Valued Queries
Bayesian Active Learning Using Arbitrary Binary Valued Queries

not useful:
Universal Rates for Interactive Learning
Interactive Machine Learning
Decision-Centric Active Learning of  Binary-Outcome Models
ActiveHedge: Hedge meets Active Learning The Power of Comparisons for Actively Learning Linear Classifiers
Active Learning for Contextual Search with Binary Feedback
Generalized Binary Search
Provable Safe Reinforcement Learning with Binary Feedback
Breaking the interactive bottleneck in multi-class classification with active selection and binary feedback
Robust active learning with binary responses



binary cls datasets:
- sklearn:
    - breast cancer https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
- tf:
    - cats_vs_dogs (images)
    - titanic (tabular)
    - yes_no (audio)
    - bool_q (nlp)
    - imdb_reviews (nlp)
    - sentiment140 (if filtering out neutral tweets, nlp)
