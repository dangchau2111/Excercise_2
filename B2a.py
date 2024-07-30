import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    # Calculate Euclidean distance between anchor and positive sample
    pos_dist = np.sum((anchor - positive) ** 2)
    
    # Calculate Euclidean distance between anchor and negative sample
    neg_dist = np.sum((anchor - negative) ** 2)
    
    # Calculate the Triplet Loss value
    loss = np.maximum(pos_dist - neg_dist + margin, 0)
    
    return loss

# Example usage
anchor = np.array([1.0, 1.0])
positive = np.array([1.1, 1.1])
negative = np.array([2.0, 2.0])

# Calculate Triplet Loss with margin = 1.0
loss = triplet_loss(anchor, positive, negative, margin=1.0)
print(f'Triplet Loss: {loss}')
