# Excercise_2

# Triplet Loss Calculation

This repository contains Python code for calculating the Triplet Loss, which is commonly used in training deep learning models for tasks such as face recognition or metric learning. The Triplet Loss function helps ensure that an anchor sample is closer to positive samples of the same class than to negative samples from different classes by a certain margin.

## Code Explanation

### B2a: Single Positive and Negative Sample

The first code snippet implements the Triplet Loss calculation for a single positive and negative sample. The formula for Triplet Loss is:

Loss=max \text{Loss} = \max(\text{pos\_dist} - \text{neg\_dist} + \text{margin}, 0) 

- `pos_dist` is the squared Euclidean distance between the anchor and positive sample.
- `neg_dist` is the squared Euclidean distance between the anchor and negative sample.
- `margin` is a hyperparameter that enforces a minimum separation between positive and negative distances.

Example usage demonstrates how to calculate the Triplet Loss with specific values for anchor, positive, and negative samples.

### B2b: Multiple Positive and Negative Samples

The second code snippet extends the Triplet Loss calculation to handle multiple positive and negative samples. For each positive sample, it computes the distance to each negative sample and sums up the losses.

- `positives` is a list of positive vectors.
- `negatives` is a list of negative vectors.
- For each pair of positive and negative samples, the loss is calculated and accumulated.

## Usage

To use the provided functions, simply call them with appropriate numpy arrays representing anchor, positive, and negative samples. Adjust the `margin` parameter as needed to control the separation margin between positive and negative pairs.

## Example

The provided example code demonstrates how to use both implementations of the Triplet Loss function. Ensure to replace the sample data with your own vectors as needed.

## Requirements

- numpy

Install the required packages using pip:

```bash
pip install numpy
