def calculate_brier_score(y_predicted=None, y_true=None,
                          sample_weight=None,
                          positive_class_label=None):
    """
    Wrapper for sklearn `brier_score_loss` function; use to calibrate
    probas from classifiers.
    """
    score = brier_score_loss(y_true=None,
                             y_prob=None,
                             sample_weight=None,
                             pos_label=1)
    return score