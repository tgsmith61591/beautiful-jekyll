---
layout: post
title: An intro to dummy encoding with Skoot
subtitle: Using Skoot to accelerate your ML pre-processing workflow
gh-repo: tgsmith61591/skoot
gh-badge: [star, fork, follow]
tags: [skoot, machine-learning, dummy-encoding]
---

This post will introduce you to dummy coding in [skoot](https://github.com/tgsmith61591/skoot), one of my projects dedicated to helping machine learning practitioners automate as much of their workflow as possible. Those who have worked in the field for a while know that 80 - 90% of a data scientist's time is spent solely on cleaning up data or building bespoke transformers to fit into an eventual production pipeline&mdash;skoot aims to solve exactly this problem by abstracting common transformer classes and data cleansing tasks into a reusable API.

**Note that this is a very high level intro to the package and that the full package documentation is available for review [here](https://tgsmith61591.github.io/skoot/)**

## Mo' data, mo' problems

*(Kinda and not to say you'd ever ask for less data. But you know what I'm getting at...)* 

Imagine a client comes at you with a business question and hands you all the data you'll need to solve it. Is it ever sparkling clean and free of errors (typographical, erroneous sensor values, data omission or other)? 

**NO!** Even when the data has been used for modeling before, you'll generally spend a significant amount of time cleaning your data, and the more features you have, the more time you'll spend on data cleansing tasks.

Let's say you're given the following dataset (the "adult data set" [available on the UCI repo](https://archive.ics.uci.edu/ml/datasets/Adult); ~3.8MB):

<div class="table-wrapper" markdown="block">

| age | workclass | fnlwgt | education | education-num | marital-status | occupation | relationship | race | sex | capital-gain | capital-loss | hours-per-week | native-country | target |
| :-- | :-------- | :----- | :-------- | :------------ | :------------- | :--------- | :----------- | :--- | :-- | :----------- | :----------- | :------------- | :------------- | :----- |
| 39 | State-gov | 77516 | Bachelors | 13 | Never-married | Adm-clerical | Not-in-family | White | Male | 2174 | 0 | 40 | United-States | <=50k |
| 50 | Self-emp-not-inc | 83311 | Bachelors | 13 | Married-civ-spouse | Exec-managerial | Husband | White | Male | 0 | 0 | 13 | United-States | <=50K |
| 38 | Private | 215646 | HS-grad | 9 | Divorced | Handlers-cleaners | Not-in-family | White | Male | 0 | 0 | 40 | United-States | <=50K |
| 53 | Private | 234721 | 11th | 7 | Married-civ-spouse | Handlers-cleaners | Husband | Black | Male | 0 | 0 | 40 | United-States | <=50K |

</div>

Our aim in this dataset is to predict whether a person makes less than or greater than $50k (binary classification). It's immediately recognizable that there are several different datatypes that will require transformations for us to be able to perform any modeling. Typically, a data scientist would spend an immense amount of time on cleaning up data and preparing meaningful features for modeling. With skoot, we can begin to chip away at this bottleneck in a matter of minutes.

### Converting categorical fields to numeric fields

If you want the cleanest pipeline possible, you'll end up building several custom `TransformerMixin` classes over the course of your modeling, one of which typically handles categorical encoding and dummy variables. There are a number of solutions to this problem out there, including the `pd.get_dummies`, but not all of them account for two issues that Skoot does:

  * What happens if there are unknown levels in the test data?
  * How can we avoid the [dummy variable trap?](http://www.algosome.com/articles/dummy-variable-trap-regression.html)

Skoot addresses these for us seamlessly. If we look at the dtypes of the dataset, we can identify which will need dummy-encoding:

{% highlight python linenos %}
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("~/Downloads/adult.data.txt", header=None,
                 names=["age", "workclass", "fnlwgt", "education", 
                        "education-num", "marital-status", 
                        "occupation", "relationship", "race", 
                        "sex", "capital-gain", "capital-loss", 
                        "hours-per-week", "native-country", "target"])
y = df.pop("target")
object_cols = df.select_dtypes(["object", 
                                "category"]).columns.tolist()

# with some examination we can see that "education-num" is just 
# an ordinal mirror of "education", so we can drop it
df.drop("education-num", axis=1, inplace=True)

# As always, we need to split our data
X_train, X_test, y_train, y_test = train_test_split(df, y, 
                                                    test_size=0.2,
                                                    random_state=42)
{% endhighlight %}

This gives us the following fields as "object" (or string) type:

* workclass
* education
* marital-status
* occupation
* relationship
* race
* sex
* native-country

With skoot we can very quickly one-hot encode all the categorical variables and drop one level (to avoid the dummy trap). Note that skoot does not force types when defining the `DummyEncoder`&mdash;this is because often times `int` fields are actually ordinal categorical features that should be encoded (like the "education-num" above). Instead, skoot allows us to define which specific columns on which to apply a transformation: 

{% highlight python linenos %}
from skoot.preprocessing import DummyEncoder

encoder = DummyEncoder(cols=object_cols, drop_one_level=True)
encoder.fit_transform(X_train).head()
{% endhighlight %}

And now our matrix looks like this:

<div class="table-wrapper" markdown="block">

| age | fnlwgt | capital-gain | capital-loss | hours-per-week | workclass_ ? | workclass_ Federal-gov | workclass_ Local-gov | workclass_ Never-worked | workclass_ Private | workclass_ Self-emp-inc | workclass_ Self-emp-not-inc | workclass_ State-gov | education_ 10th | education_ 11th | education_ 12th | education_ 1st-4th | education_ 5th-6th | education_ 7th-8th | education_ 9th | education_ Assoc-acdm | education_ Assoc-voc | education_ Bachelors | education_ Doctorate | education_ HS-grad | education_ Masters | education_ Preschool | education_ Prof-school | marital-status_ Divorced | marital-status_ Married-AF-spouse | marital-status_ Married-civ-spouse | marital-status_ Married-spouse-absent | marital-status_ Never-married | marital-status_ Separated | occupation_ ? | occupation_ Adm-clerical | occupation_ Armed-Forces | occupation_ Craft-repair | occupation_ Exec-managerial | occupation_ Farming-fishing | occupation_ Handlers-cleaners | occupation_ Machine-op-inspct | occupation_ Other-service | occupation_ Priv-house-serv | occupation_ Prof-specialty | occupation_ Protective-serv | occupation_ Sales | occupation_ Tech-support | relationship_ Husband | relationship_ Not-in-family | relationship_ Other-relative | relationship_ Own-child | relationship_ Unmarried | race_ Amer-Indian-Eskimo | race_ Asian-Pac-Islander | race_ Black | race_ Other | sex_ Female | native-country_ ? | native-country_ Cambodia | native-country_ Canada | native-country_ China | native-country_ Columbia | native-country_ Cuba | native-country_ Dominican-Republic | native-country_ Ecuador | native-country_ El-Salvador | native-country_ England | native-country_ France | native-country_ Germany | native-country_ Greece | native-country_ Guatemala | native-country_ Haiti | native-country_ Holand-Netherlands | native-country_ Honduras | native-country_ Hong | native-country_ Hungary | native-country_ India | native-country_ Iran | native-country_ Ireland | native-country_ Italy | native-country_ Jamaica | native-country_ Japan | native-country_ Laos | native-country_ Mexico | native-country_ Nicaragua | native-country_ Outlying-US(Guam-USVI-etc) | native-country_ Peru | native-country_ Philippines | native-country_ Poland | native-country_ Portugal | native-country_ Puerto-Rico | native-country_ Scotland | native-country_ South | native-country_ Taiwan | native-country_ Thailand | native-country_ Trinadad&Tobago | native-country_ United-States | native-country_ Vietnam |
| :-- | :----- | :----------- | :----------- | :------------- | :----------- | :--------------------- | :------------------- | :---------------------- | :----------------- | :---------------------- | :-------------------------- | :------------------- | :-------------- | :-------------- | :-------------- | :----------------- | :----------------- | :----------------- | :------------- | :-------------------- | :------------------- | :------------------- | :------------------- | :----------------- | :----------------- | :------------------- | :--------------------- | :----------------------- | :-------------------------------- | :--------------------------------- | :------------------------------------ | :---------------------------- | :------------------------ | :------------ | :----------------------- | :----------------------- | :----------------------- | :-------------------------- | :-------------------------- | :---------------------------- | :---------------------------- | :------------------------ | :-------------------------- | :------------------------- | :-------------------------- | :---------------- | :----------------------- | :-------------------- | :-------------------------- | :--------------------------- | :---------------------- | :---------------------- | :----------------------- | :----------------------- | :---------- | :---------- | :---------- | :---------------- | :----------------------- | :--------------------- | :-------------------- | :----------------------- | :------------------- | :--------------------------------- | :---------------------- | :-------------------------- | :---------------------- | :--------------------- | :---------------------- | :--------------------- | :------------------------ | :-------------------- | :--------------------------------- | :----------------------- | :------------------- | :---------------------- | :-------------------- | :------------------- | :---------------------- | :-------------------- | :---------------------- | :-------------------- | :------------------- | :--------------------- | :------------------------ | :----------------------------------------- | :------------------- | :-------------------------- | :--------------------- | :----------------------- | :-------------------------- | :----------------------- | :-------------------- | :--------------------- | :----------------------- | :------------------------------ | :---------------------------- | :---------------------- |
| 39.0 | 77516.0 | 2174.0 | 0.0 | 40.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 |
| 50.0 | 83311.0 | 0.0 | 0.0 | 13.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 |
| 38.0 | 215646.0 | 0.0 | 0.0 | 40.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 |
| 53.0 | 234721.0 | 0.0 | 0.0 | 40.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 |
| 28.0 | 338409.0 | 0.0 | 0.0 | 40.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

</div>

To apply this to your test data, just as with any other scikit-learn transformer, you simply use the `transform` method:


{% highlight python linenos %}
encoder.transform(X_test)
{% endhighlight %}


### Things to note

* The resulting features drop one factor level from each categorical variable if `drop_one_level=True` is specified (by default).
* We address the situation where an unknown factor level is present

Here's a demo of what happens when there's a new factor level present:


{% highlight python linenos %}
# select a test row:
test_row = X_test.iloc[0]

# set the country to something that is obviously not real:
test_row.set_value('native-country', "Atlantis")

# transform the new row:
trans2 = encoder.transform(pd.DataFrame([test_row]))

# prove that we did not assign a country encoding:
nc_mask = trans2.columns.str.contains("native-country")
assert trans2[trans2.columns[nc_mask]].sum().sum() == 0
{% endhighlight %}


And there you have it! <2 minutes to dummy encode your categorical features. The full code for this example is located in the [code folder](https://github.com/tgsmith61591/tgsmith61591.github.io/blob/master/code/2018-06-18-intro-to-skoot-dummy.ipynb).

**Questions? Technical remarks? Feel free to email me at taylor.smith@alkaline-ml.com**
