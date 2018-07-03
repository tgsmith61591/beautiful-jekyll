---
layout: post
title: Conda envs in Pyspark
subtitle: 3 reasons you should be deploying your Conda environments for your Pyspark jobs
tags: [python, pyspark, tutorials]
---

If you've only ever tinkered with Hadoop within the context of a [sandbox](https://hortonworks.com/products/sandbox/), you may never have encountered one of the inevitabililities of Enterprise-scale distributed computing: **different machines have different configurations**. Even when synchronized with tools such as [Puppet](https://puppet.com), datanodes in a Hadoop cluster may not be a mirror image of edgenodes. Especially in the (relatively common) case that you've developed a custom python package you'd like to use across a cluster.

## Conda on the cluster

For local python development, [Anaconda](https://www.anaconda.com) exists to manage & modularize your dependencies and environments. However, for a software package as devoted as it is to environment management, the documentation that exists around using Conda environments in a cluster is sparse at best. 

In this tutorial, we'll cover how we can manage our environments across the cluster specifically for use in our Pyspark jobs. Though there are several methods to distributing your code to your datanodes, shipping a Conda environment is likely to be your most robust option for the following reasons.


### Reason 1: your package doesn't exist on the cluster

If you've developed a custom python package, it's unlikely it exists across all the executors on the cluster. Moreover, if it *does* and you make changes, syncing a production cluster via Puppet is rarely going to be the best option. 

Let's imagine we have a huge 2D matrix, and we want to compute a given percentile for each feature using Pandas. We can do this in a distributed fashion on each executor by parallelizing the transpose of our matrix:

{% highlight python linenos %}
import numpy as np

# Create a large random array
random_state = np.random.RandomState(42)
X = random_state.rand(1000000, 100)

# Parallelize into an RDD
rdd = sc.parallelize(X.T.tolist(), 8)

# This function will compute a different percentile depending on
# the group value
def n_tile(x, q):
    import pandas as pd
    return pd.qcut(x, q=q)

# Map the function over each column
decile_intervals = rdd.map(lambda x: n_tile(x, 10)).collect()
{% endhighlight %}

If you're operating in a cluster, what you'll likely find when you try to collect your result is that you'll encounter an `ImportError` on your executors:


```
  File "/path/to/spark/lib/spark/python/pyspark/worker.py", line 111, in main
    process()
  File "/path/to/spark/lib/spark/python/pyspark/worker.py", line 106, in process
    serializer.dump_stream(func(split_index, iterator), outfile)
  File "/path/to/spark/lib/spark/python/pyspark/serializers.py", line 263, in dump_stream
    vs = list(itertools.islice(iterator, batch))
  File "<stdin>", line 2, in pandas
ImportError: No module named pandas

	at org.apache.spark.api.python.PythonRunner$$anon$1.read(PythonRDD.scala:166)
	at org.apache.spark.api.python.PythonRunner$$anon$1.<init>(PythonRDD.scala:207)
	at org.apache.spark.api.python.PythonRunner.compute(PythonRDD.scala:125)
	at org.apache.spark.api.python.PythonRDD.compute(PythonRDD.scala:70)
	at org.apache.spark.rdd.RDD.computeOrReadCheckpoint(RDD.scala:306)
	at org.apache.spark.rdd.RDD.iterator(RDD.scala:270)
	at org.apache.spark.scheduler.ResultTask.runTask(ResultTask.scala:66)
	at org.apache.spark.scheduler.Task.run(Task.scala:89)
	at org.apache.spark.executor.Executor$TaskRunner.run(Executor.scala:242)
	at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1152)
	at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:622)
	at java.lang.Thread.run(Thread.java:745)
```

This is because Pandas is a non-standard Python package and it's unlikely to be installed on an executor be default. This is going to be the case if you're running custom package code across Spark datanodes.


### Reason 2: you may not have the permissions you think you do

Distributing an Anaconda environment is not the only way to get your code to executors. In fact, it's not even the easiest or default method. Spark makes it incredibly easy for you to distribute an `.egg` file in your `spark-submit` statement:

```bash
$ spark-submit --py-files my_egg.egg my_script.py
```

However, in many production environments, you cannot predict what permissions you will have on executors and can easily encounter permissions issues:

```
The following error occurred while trying to extract file(s) to the Python egg
cache:

  [Errno 13] Permission denied: '/home/.python-eggs'

The Python egg cache directory is currently set to:

  /home/.python-eggs
```

And anyone who has worked on an enterprise cluster before knows that getting permissions amended on production clusters is a loooong process.


### Reason 3: your package contains code that needs compiling

Even if you *do* have permission to the directory, if your python package contains C code, you're at the mercy of the versions of numpy, scipy or other C-based python packages located on the executors as to whether your code will work. If the version you built under does not match that on the executors, you can always hit low level dtype errors:

```
ValueError: numpy.dtype has the wrong size, try recompiling. Expected 88, got 96

    at org.apache.spark.api.python.PythonRunner$$anon$1.read(PythonRDD.scala:166)
    at org.apache.spark.api.python.PythonRunner$$anon$1.<init>(PythonRDD.scala:207)
    at org.apache.spark.api.python.PythonRunner.compute(PythonRDD.scala:125)
    at org.apache.spark.api.python.PythonRDD.compute(PythonRDD.scala:70)
    at org.apache.spark.rdd.RDD.computeOrReadCheckpoint(RDD.scala:306)
    at org.apache.spark.CacheManager.getOrCompute(CacheManager.scala:69)
    at org.apache.spark.rdd.RDD.iterator(RDD.scala:268)
    at org.apache.spark.api.python.PythonRDD.compute(PythonRDD.scala:70)
    at org.apache.spark.rdd.RDD.computeOrReadCheckpoint(RDD.scala:306)
    at org.apache.spark.rdd.RDD.iterator(RDD.scala:270)
    at org.apache.spark.scheduler.ResultTask.runTask(ResultTask.scala:66)
    at org.apache.spark.scheduler.Task.run(Task.scala:89)
    at org.apache.spark.executor.Executor$TaskRunner.run(Executor.scala:214)
    at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1152)
    at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:622)
    ... 1 more
```

## Distributing your entire Conda environment: a more sustainable deployment pattern

We can address all of these problems in one fell swoop by simply shipping our entire environment to the datanodes. There are (at least) two methods we can use for this.

### Method 1: installing amongst your global Anaconda envs

First, create your anaconda environment:

```bash
$ conda create -n my-global-env --copy -y python=3.5 numpy scipy pandas
```

Once you’ve created your Conda environment, you'll install your custom python package inside of it (if necessary):

```bash
$ source activate my-global-env
(my-global-env) $ python setup.py install
```

After you’ve built and installed your package into your environment, it's ready to be zipped and shipped. If your environment is active (sourced), you can find where it’s installed with the following command:

```bash
(my-global-env) $ which conda
//anaconda/envs/my-global-env/bin/conda
```

We are going to `cd` into the `envs` directory, zip up the environment and prepare it for shipping (assuming we want to launch pyspark shell from your home dir):

```bash
(my-global-env) $ cd /anaconda/envs
(my-global-env) $ zip -r my-global-env.zip my-global-env/
(my-global-env) $ mv my-globa-env.zip ~/
```

Now we need to symlink your conda env:

```bash
(my-global-env) $ cd ~/ && mkdir MYGLOBALENV
(my-global-env) $ cd MYGLOBALENV/ && ln -s /anaconda/envs/my-global-env/ my-global-env
(my-global-env) $ cd ..
(my-global-env) $ export PYSPARK_PYTHON=./MYGLOBALENV/my-global-env/bin/python
```

To make this environment available to the executors, there are two steps we need to take:
1. Distribute the package
2. Change the default python for Pyspark to this location (we just handled that with the export)


The variable that controls the python environment in Spark is named `PYSPARK_PYTHON` and is set before calling `pyspark` or `spark-submit`. Here’s how you can start pyspark with your anaconda environment (feel free to add other Spark conf args, etc.):

```bash
(my-global-env) $ PYSPARK_PYTHON=./MYGLOBALENV/my-global-env/bin/python pyspark \
                  --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./MYGLOBALENV/my-global-env/bin/python \
                  --master yarn-client \
                  --archives my-global-env.zip#MYGLOBALENV
```

Test that it worked by importing a non-standard version of a package in your environment across the cluster:

{% highlight python linenos %}
def npv(x):
    import numpy as np
    return np.__version__

set(sc.parallelize(range(1000), 10).map(npv).collect())
{% endhighlight %}

You should only have one version in your `set`.


### Method 2: shipping a local Anaconda environment (my preferred method)

In this method, we'll create an Anaconda environment within the same directory from which we will be deploying our application (or launching our shell). First, create and source your local anaconda environment (notice the new flags in the `conda create` statement, and the `pwd` in the activation statement):

```bash
$ conda create -m -p my-local-env --yes python=3.5 numpy scipy pandas
$ source activate `pwd`/my-local-env
```

Install your python package inside of it (if necessary):

```bash
(my-local-env) $ python setup.py install
```

Zip your environment (can be done from a sourced or deactivated environment):

```bash
(my-local-env) $ zip -r my-local-env.zip my-local-env
```

With our environment sourced, we can launch our application as follows. I find it's more modular to export the name of the environment to be more flexible:

```bash
(my-local-env) $ export CONDAENV=my-local-env
(my-local-env) $ PYSPARK_DRIVER_PYTHON=`which python` \
                 PYSPARK_PYTHON=./${CONDAENV}_zip/${CONDAENV}/bin/python pyspark \
                 --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./${CONDAENV}_zip/${CONDAENV}/bin/python \
                 --archives "./${CONDAENV}.zip#${CONDAENV}_zip"
```

You can test that this worked in the same manner as above.


## Epilogue

And that's it! Two approaches to a modular & sustainable Pyspark cluster deployment pattern. By shipping anaconda environments, you can avoid permissions errors, version mismatch problems, and other cluster management woes. 


**Questions? Technical remarks? Feel free to email me at taylor.smith@alkaline-ml.com, or leave a comment below**
