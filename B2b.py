import numpy as np

def triplet_loss(anchor, positives, negatives, margin=1.0):
    """
    Calculate the Triplet Loss with multiple positive and negative samples.

    Parameters:
    - anchor: The anchor vector (numpy array)
    - positives: List of positive vectors (numpy array)
    - negatives: List of negative vectors (numpy array)
    - margin: The margin value (float)

    Returns:
    - loss: The computed triplet loss (float)
    """
    # Initialize the loss
    loss = 0

    # Convert positives and negatives to numpy arrays if they are lists
    positives = np.array(positives)
    negatives = np.array(negatives)

    # Calculate distances
    pos_distances = np.sum((anchor - positives) ** 2, axis=1)  # Shape: (number of positives,)
    neg_distances = np.sum((anchor - negatives) ** 2, axis=1)  # Shape: (number of negatives,)

    # Compute loss for each positive-negative pair
    for pos_dist in pos_distances:
        for neg_dist in neg_distances:
            loss += np.maximum(0, pos_dist - neg_dist + margin)
    
    return loss

# Ví dụ sử dụng
anchor = np.array([1.0, 2.0])
positives = [np.array([1.1, 2.1]), np.array([0.9, 1.9])]
negatives = [np.array([3.0, 3.0]), np.array([4.0, 4.0]), np.array([5.0, 5.0]), np.array([6.0, 6.0]), np.array([7.0, 7.0])]

# Tính Triplet Loss với margin = 1.0
loss = triplet_loss(anchor, positives, negatives, margin=1.0)
print(f'Triplet Loss: {loss}')
