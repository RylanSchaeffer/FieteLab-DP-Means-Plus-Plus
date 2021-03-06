import numpy as np
from sklearn.metrics import adjusted_rand_score, rand_score, adjusted_mutual_info_score, \
    normalized_mutual_info_score
from typing import Dict, Tuple


def compute_predicted_clusters_scores(cluster_assignment_posteriors: np.ndarray,
                                      true_cluster_assignments: np.ndarray,
                                      ) -> Tuple[Dict[str, float], np.ndarray]:
    # cluster assignment posteriors has shape (number obs, num clusters)
    # (r, c)th element is probability the rth observation belongs to cth cluster
    # true_cluster_labels: integer classes with shape (num obs, )

    if len(cluster_assignment_posteriors.shape) == 2:
        pred_cluster_labels = np.argmax(cluster_assignment_posteriors,
                                        axis=1)
    elif len(cluster_assignment_posteriors.shape) == 1:
        pred_cluster_labels = cluster_assignment_posteriors
    else:
        raise ValueError('Wrong number of dimensions.')

    rnd_score = rand_score(labels_pred=pred_cluster_labels,
                           labels_true=true_cluster_assignments)

    adj_rnd_score = adjusted_rand_score(labels_pred=pred_cluster_labels,
                                        labels_true=true_cluster_assignments)

    adj_mut_inf_score = adjusted_mutual_info_score(labels_pred=pred_cluster_labels,
                                                   labels_true=true_cluster_assignments)

    norm_mut_inf_score = normalized_mutual_info_score(labels_pred=pred_cluster_labels,
                                                      labels_true=true_cluster_assignments)

    scores_results = {
        'Rand Score': rnd_score,
        'Adjusted Rand Score': adj_rnd_score,
        'Adjusted Mutual Info Score': adj_mut_inf_score,
        'Normalized Mutual Info Score': norm_mut_inf_score,
    }

    return scores_results, pred_cluster_labels

