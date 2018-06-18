---
layout: post
title: Skoot&mdash;an intro
subtitle: Using Skoot to accelerate your ML workflow
gh-repo: tgsmith61591/skoot
gh-badge: [star, fork, follow]
tags: [test]
---

This post will introduce you to [skoot](https://github.com/tgsmith61591/skoot), one of my projects dedicated to helping machine learning practioners automate as much of their workflow as possible. Those who have worked in the field for a while know that 80 - 90% of a data scientist's time is spent solely on cleaning up data or building bespoke transformers to fit into an eventual production pipeline&mdash;skoot aims to solve exactly this problem by abstracting common transformer classes and data cleansing tasks into a reusable API.

**Note that this is just an intro to the package and that the full package documentation is available for review [here](https://tgsmith61591.github.io/skoot/)**

## Mo' data, mo' problems

Kinda (and not to say I'd ever ask for less data). But you know what I'm getting at... imagine a client comes at you with a business question and hands you all the data you'll need to solve it. Is it ever sparkling clean and free of errors (typographical, erroneous sensor value, omission or other)? **NO!** This is such a given it's nearly a cliché to even bring up here, but the fact of the matter is you will always spend a significant amount of time cleaning your data, and the more features you have, the more time you'll spend on data cleansing tasks.

Let's say you're given the following dataset (the 2016 green taxi trip data [available here](https://data.cityofnewyork.us/Transportation/2016-Green-Taxi-Trip-Data/hvrh-b6nb); ~MB):



It's immediately recognizable that there are several different datatypes that will require transformations for us to be able to perform any modeling. Typically, a data scientist would spend an immense amount of time on cleaning up data and preparing meaningful features for modeling. With skoot, we can begin to chip away at this bottleneck.

### Convert datetime fields

If you want the cleanest pipeline possible, you'll end up building several custom `TransformerMixin` classes over the course of your modeling, one of which typically handles datetime conversions and all the peculiarities that come along with handling datetimes. Skoot automates this for us:

{% highlight python linenos %}
import pandas as pd
from skoot.preprocessing import DateTransformer

data = pd.read_csv("~/path/to/2016_Green_Taxi_Trip_Data.csv")
date_trans = DateTransformer(cols=["lpep_pickup_datetime", "Lpep_dropoff_datetime"],
                             date_format="%Y %b %d TODO")
{% endhighlight %}



Here's a useless table:

| Number | Next number | Previous number |
| :------ |:--- | :--- |
| Five | Six | Four |
| Ten | Eleven | Nine |
| Seven | Eight | Six |
| Two | Three | One |


How about a yummy crepe?

![Crepe](http://s3-media3.fl.yelpcdn.com/bphoto/cQ1Yoa75m2yUFFbY2xwuqw/348s.jpg)

Here's a code chunk:

~~~
var foo = function(x) {
  return(x + 5);
}
foo(3)
~~~

And here is the same code with syntax highlighting:

```javascript
var foo = function(x) {
  return(x + 5);
}
foo(3)
```

And here is the same code yet again but with line numbers:

{% highlight javascript linenos %}
var foo = function(x) {
  return(x + 5);
}
foo(3)
{% endhighlight %}

## Boxes
You can add notification, warning and error boxes like this:

### Notification

{: .box-note}
**Note:** This is a notification box.

### Warning

{: .box-warning}
**Warning:** This is a warning box.

### Error

{: .box-error}
**Error:** This is an error box.
