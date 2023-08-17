# Adaptive Knowledge-Graph based Recommender with User's Cross-Item-Category Exploration Factor

## Overview

In recommendation systems, user preferences for item categories influence choices and can profile different users. Several users enjoy exploring various categories of items in recommendations, while others prefer specific categories. To fully utilize user preference for categories, we construct two novel Knowledge Graphs (KGs): a User-Item KG (KG-UI) that captures interactions between users and items, and a User-Category KG (KG-UC) that directly models user preference for explored categories. Furthermore, we propose a novel recommendation system, knowledge graph-based recommendation system with user’s cross-item-category exploration factor (KCEF). KCEF effectively captures user preferences for categories and enhances recommendation performance. Specifically, we design an information aggregation framework aggregating both KGs to learn entity representations jointly. Moreover, it explores independence and correlations among categories in users’ historical purchasing logs, which benefits the user preference learning. In addition, we convert user preferences for unexplored item categories into a Cross-item-category Exploration Factor (CEF) and establish a category-wise loss function for recommendation. We propose a negative sampling based on the category for an end-toend framework to optimize the above loss function for consistency. Experimental results on three benchmark datasets demonstrate that KCEF significantly improves over the stateof- the-art methods.

Please see the paper for more details.

This is our PyTorch implementation for the paper:

## Environment Requirement

The code has been tested running under Python 3.6.5. The required packages are as follows:

-- pytorch == 1.8.1
-- numpy == 1.19.5
-- scipy == 1.1.0
-- sklearn == 0.20.0
-- torch_scatter == 2.0.9
-- networkx == 2.5

## Dataset

We provide three processed datasets: Movielens, Last-FM, and Alibaba-iFashion.

## Example to Run the Codes

You can run the .py file directly, using the following command to run the model:

$ python main.py
