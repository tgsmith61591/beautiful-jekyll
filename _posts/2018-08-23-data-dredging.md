---
layout: post
title: How to ensure model obsolescence (part 2)
subtitle: Data dredging
tags: [python, machine-learning, scikit-learn, best-practices, interview-prep, tutorials]
---

In the spirit of my last post, I want to continue talking about some common mistakes I see among machine learning practitioners. Last time, we saw how covariate shift can be accidentally introduced by (seemingly harmlessly) applying a `fit_transform` to your test data. This time, I want to cover an equally egregious practice: __data dredging__.


## What's data dredging?

Also commonly called "p-hacking," data dredging is essentially the practice of allowing your test or validation set to inform decisions around your model-building or hyper-parameter tuning. In a practical sense, it's when you repeatedly expose your holdout set to your model while continuing to make adjustments.


## When does it typically occur?

Most of the time I see data dredging, it's in the context of evaluating a grid search. Here's a quick problem setup:


{% highlight python linenos %}
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import randint
import numpy as np

# Load the data
X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Define (roughly) our hyper parameters
hyper = {
    'max_depth': randint(3, 10),
    'n_estimators': randint(25, 250),
    'learning_rate': np.linspace(0.001, 0.01, 20),
    'min_samples_leaf': [1, 5, 10]
}

# Define our CV class (remember to always shuffle!)
cv = KFold(shuffle=True, n_splits=3, random_state=1)

# Define our estimator
search = RandomizedSearchCV(GradientBoostingRegressor(random_state=42),
                            scoring='neg_mean_squared_error', n_iter=25, 
                            param_distributions=hyper, cv=cv,
                            random_state=12, n_jobs=4)

# Fit the grid search
search.fit(X_train, y_train)
{% endhighlight %}


At this point, we've fit a valid model, and we want to know how it performs. So what do most people do? They score the model against their holdout set:

{% highlight python linenos %}
from sklearn.metrics import mean_squared_error

# Evaluate:
print("Test MSE: %.3f" % mean_squared_error(y_test, search.predict(X_test)))
{% endhighlight %}

__But this is a *dangerous* practice!!!__ By introducing your holdout set too early, your design decisions may reflect what you've learned about the model's performance. Maybe you re-fit, trying more estimators or a steeper learning rate. In both cases, you only did so because your model didn't perform well enough against the holdout set, and what ends up happening is that you begin to slowly tailor the model until it scores well-enough against your test set.

In a sense, you end up inadvertently fitting your test set. __Any model-tuning action you take as a result of scoring against your holdout set is data dredging.__


## "But I'm not fitting my test set. How is this bad?"

You don't have to fit your test set directly in order to leak information to your models. There is an inherit bias that the analyst/scientist/engineer imparts on his/her models, and by making repeated decisions based on hold-out performance, you inflate the personal bias you impose. 

You might end up acheiving outstanding performance on your test set, but your model will likely not generalize well to the new data it sees once you deploy it.


## Why does this happen?

I have several hypotheses for why I see this practice so much:

#### 1. Academic settings don't emphasize the model lifecycle

In most academic settings, machine learning problems begin and end with achieving the best model possible on the provided test set. Naturally, if you're already provided said holdout set, you're more tempted to p-hack, and budding data scientists don't learn to consider that the model will have to make decisions on future data that does *not* have corresponding labels.

#### 2. People misunderstand the purpose of cross-validation

For such a well-documented concept, few things in ML share the level of misunderstanding that CV suffers. Lots of folks use it interchangeably with train/test splits, and yet others seem to assume that if a model is fit with cross-validation, it's impervious to the perils of overfitting. 

Neither is true, and as a result, CV isn't used to its full potential.


## How can we avoid it?

Avoiding data dredging is more simple that you might think. The answer, as with most things in machine learning, is cross validation! We can approach this several ways.

### Benchmark a model with ``cross_val_score``

You typically don't go straight into a grid search. First, you try several models. Scikit allows us to fit a model in the context of cross validation and examine the fold scores. This is useful for determining whether a model will perform in the ballpark of business requirements before a lengthy tuning process:

{% highlight python linenos %}
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# Set our CV seed
cv = KFold(n_splits=3, random_state=0, shuffle=True)

# Fit and score a model in CV:
cross_val_score(GradientBoostingRegressor(random_state=42),
                X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
# Returns: array([ -7.62352454, -15.10931642, -16.47872053])
{% endhighlight %}

If your model needs to perform at, say, an MSE of <= 15 (for the arbitrary sake of argument), you might feel confident that you could tune a model that will perform to standards. However, if you need to achieve, say, <= 5, you may need to select a different model family or tune your hyper parameters significantly.

### Extract the CV scores from the grid search

If you've already fit your search, it's even easier. Scikit allows us to extract the cross validation scores from a grid search without us having to touch the test set:

{% highlight python linenos %}
import pandas as pd

pd.DataFrame(search.cv_results_)\
  .sort_values('mean_test_score',
               # descend since neg MSE
               ascending=False)\
  .head()
{% endhighlight %}

This gives us a pandas frame from which we ascertain the CV scores for each fold, the parameters that contribute to the highest scores, and other valuable information: 

|   | split0_test_score  | split1_test_score  | split2_test_score  | ... |  params |
|---|--------------------|--------------------|--------------------|-----|---------|
| 14|          -9.779118 |         -36.088421 |         -11.244133 | ... | {'learning_rate': 0.01, 'max_depth': 7, 'min_s... |
|  2|         -13.972549 |         -38.821430 |         -15.160443 | ... | {'learning_rate': 0.009052631578947368, 'max_d... |
| 13|         -14.944225 |         -39.504609 |         -15.895012 | ... | {'learning_rate': 0.0062105263157894745, 'max_... |


## Closing thoughts

You need to be careful about introducing your test set to the mix of things. The test set is generally intended to aid in __model selection__, and should be introduced as a selection technique *after* competing models have been tuned.

Rather that using your test set to gauge generalizability, use cross-validation to make informed decisions about how to further tune a model's parameters. The full code for this example is located in the [code folder](https://github.com/tgsmith61591/tgsmith61591.github.io/blob/master/code/2018-08-23-data-dredging.ipynb).


