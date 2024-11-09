# Amazon Product Recommendation System

A deep learning-based recommendation system built to predict and recommend Amazon products to users based on predicted user ratings. This project uses a deep learning model to estimate the score (rating) a user might give to a product they haven’t purchased yet and suggests the top N products with the highest predicted ratings.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [How It Works](#how-it-works)

## Project Overview

The goal of this recommendation system is to predict user ratings for products they haven’t purchased and recommend the top N products with the highest predicted ratings. The system first encodes user and product IDs, then trains a deep learning model to learn user-product preferences based on existing ratings. After training, the model can estimate scores for products a user hasn’t bought, making it suitable for generating personalized recommendations.

## Dataset

The dataset contains user IDs, product IDs, and rating information. The essential columns are:
- `UserID`: The unique identifier of each user.
- `ProductID`: The unique identifier of each product.
- `Rating`: The score (rating) a user has given to a product.

Sample data:
```
| UserID      | ProductID     | Rating |
|-------------|---------------|--------|
| A2Y3A341VDK37H | B00HFI55N2 | 4.0    |
| A240FRPD4MEXND | B00KIMX4EY | 5.0    |
| ...         | ...           | ...    |
```

## Model Architecture

This recommendation model uses embedding layers to map user and product IDs to dense vector representations, capturing latent features of both users and products. The main idea behind the recommendation is to use the **dot product** of the user and product vectors to measure how well a particular product matches the user’s preferences.

### Dot Product Explanation

In this model, the dot product is computed between the user vector and the product vector, which serves as a measure of interaction strength between a specific user and product. Conceptually, this can be thought of as the alignment of two vectors in the latent space:

- **User Vector**: Represents the user’s preferences across different hidden dimensions (features learned by the model).
- **Product Vector**: Represents the characteristics of a product across the same dimensions.

By calculating the dot product between these two vectors, the model produces a single score that reflects how closely the product aligns with the user's preferences. A higher dot product value suggests a stronger match, while a lower value suggests a weaker match.

In mathematical terms, if **U** is the user vector and **P** is the product vector, the dot product is calculated as:

$\text{Dot Product} = U \cdot P = \sum_{i=1}^{n} U_i \times P_i$

where \( U_i \) and \( P_i \) are the components of the user and product vectors, respectively, and \( n \) is the embedding size.

This dot product serves as the input to the final layer, which uses a sigmoid activation to output a predicted rating, bounded between 0 and 1.

## How It Works

1. **Predict Initial Ratings**: The model is trained on existing user-product ratings to learn their interactions.
2. **Find Unpurchased Products**: For a given user, the model identifies products they have not rated or purchased.
3. **Predict Scores for New Products**: The model takes the encoded user-product pairs and predicts scores for each of the unpurchased products.
4. **Generate Recommendations**: The products are sorted by predicted score, and the top N items are recommended to the user.
