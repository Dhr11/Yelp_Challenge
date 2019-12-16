## Motivation
Matrix factorization makes a recommendation based on previous ratings alone. The recommendations can be improved by using the reviews data. Reviews provide much more information regarding the user sentiment towards the business than ratings. However, all reviews are not the same. Some reviews can be extremely useful, and some can provide no information at all. Not only that some reviews provide no useful information but it can be argued that these reviews can impede the model’s performance. Most of the models use reviews to build a better recommendation system consider all reviews the same and don't try to explore how useful is the review. Neural Attentional Regression model with Review-level
Explanations (NARRE) tries addressing this problem of recommendation system. NARRE works well in predicting the ratings accurately and additionally learning the importance of each review.

### file info
model_narre.ipynb  - initial "Just" model for narre
Data_creation.ipynb --  original data creation script
Data_creation_filtered.ipynb -- data creation for more filtered and less sparse version
Dataset_2_model_narre_Lesser_dropout.ipynb  -- final model+ train for dataset II, with less dropout
Dataset_2_model_narre_No_embedding_regularization.ipynb -- -- final model+ train for dataset II, with no embedding regularization
