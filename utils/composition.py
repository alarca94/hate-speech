def _average(vec1, vec2, alpha=None, beta=None, gamma=None, K=None):
    """
    Compose the vectors of two words by summing the vectors
    """
    if vec1 is None:
        return vec2
    if vec2 is None:
        return vec1

    return (vec1 + vec2) / 2

def _sum(vec1, vec2, alpha=None, beta=None, gamma=None, K=None):
    """
    Compose the vectors of two words by summing the vectors
    """
    if vec1 is None:
        return vec2
    if vec2 is None:
        return vec1

    return vec1 + vec2

def compose(elements, word2vecs, alpha=0.5, beta=0.5, gamma=1, K=15, start='left', comp_func=_sum, stopwords=[]):
    init_id = 0
    composition = None

    ids = list(range(len(elements)))
    if start == 'right':
        ids = ids[::-1]

    # Assign to the composition the vector of the first component of the sentence (compute composition if necessary)
    while composition is None and init_id < len(ids):
        if isinstance(elements[ids[init_id]], list):
            composition = compose(elements[ids[init_id]], word2vecs, alpha, beta, gamma, K, start, comp_func, stopwords)
        else:
            # Check if the current word exists in the dictionary
            if elements[ids[init_id]] in word2vecs.vocab.keys() and elements[ids[init_id]].lower() not in stopwords:
                composition = word2vecs[elements[ids[init_id]]]
        init_id += 1

    # If there are still words to consider, continue composing the sentence
    if init_id < len(ids):
        for i in ids[init_id:]:
            if isinstance(elements[i], list):
                v1 = compose(elements[i], word2vecs, alpha, beta, gamma, K, start, comp_func, stopwords)
                composition = comp_func(composition, v1, alpha, beta, gamma, K)
            else:
                if elements[i] in word2vecs.vocab.keys() and elements[i].lower() not in stopwords:
                    v1 = word2vecs[elements[i]]
                    composition = comp_func(composition, v1, alpha, beta, gamma, K)
    
    return composition