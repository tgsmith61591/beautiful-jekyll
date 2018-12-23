---
layout: post
title: Auto-deploying your Sphinx doc on gh-pages
subtitle: Automating your documentation deployments
tags: [python, tutorials, cicd, gh-pages, deployment, bear, sphinx]
---


## The issue

Documentation in the software world can be [pretty bad](https://dzone.com/articles/why-developers-write-horrible-documentation-and-ho). 
It's all too common to find a promising git project with several hundred stars and nothing but
a `README` to document package use. While tools like [sphinx](http://www.sphinx-doc.org/en/master/) have
made auto-documentation more straight-forward for developers, hosting said documentation
poses another hurdle.

When it comes to publishing & hosting documentation, Github solves the problem
with its [`gh-pages`](https://pages.github.com) functionality, which allows you to host static 
web pages from a branch. But if you've ever used it, you're probably aware of the headaches it can cause. 

In this post, I'm going to share a workflow I've developed to automate my documentation builds
and deployments to `gh-pages` using Circle CI.



## The first push to `gh-pages`:

Before we automate the builds, it's helpful to have the `gh-pages` branch on the
server. Make sure you have all of your changes committed and pushed before proceeding.

From the `master` (or your default) branch, we're going to checkout a new
[orphaned branch](https://csinaction.com/2016/02/12/orphaned-brachnes-in-git/) called
`gh-pages`. This is a magic branch name, so it's important you call it this, otherwise you'll
have to do some voodoo in your project settings to make this all work.

```bash
$ git checkout --orphan gh-pages
```

Now, we're going to delete everything. Everything *except* the `.git` directory!

This sounds scary, but if you've committed everything, it's going to be just fine! Here's a quick hack
that will delete all files except the `.git` directory:

```bash
$ find . ! -name ".git/*" -type f -exec rm -f {} +
```

But depending on whatever else you have in the project folder, you may have several other artifacts or 
directories to clean up. **Just make sure the `.git` folder goes untouched.**

The first push is easy. It's going to be an empty commit:

```bash
$ git add --all
$ git commit --allow-empty
$ git push origin gh-pages
```


## Automating all future doc deployments

Now that the branch exists on remote, we can streamline things. If you don't already have a Circle
account, head on over to the [sign-up](https://circleci.com/signup/) page and click "Sign Up with Github".
From there, you can setup your `.circleci/config.yml` add your project to a Circle workflow to build/test on
Github pushes. If you're not familiar with how to do this, follow [this tutorial](https://circleci.com/docs/2.0/tutorials/)
on getting setup with Circle.


### Creating a build script for your documentation

Here's what my [build script](https://github.com/tgsmith61591/gh_automation/blob/master/build_tools/circle/build_push_doc.sh) looks like. 
It's located inside the `build_tools/circle` directory. I've littered comments throughout
to try to explain each step:

```bash
#!/bin/bash

# Fail out on an error
set -e

# This is a hack, but we have to make sure we're only ever running this
# from the top level of the package and not in the subdirectory...
# Shouldn't ever be an issue the way we've got this setup, and you'll
# want to change it a bit to make it work with your project structure.
if [[ ! -f gh_doc_automation/__init__.py ]]; then
    echo "This must be run from the gh_doc_automation project directory"
    exit 3
fi

# CD into docs, make them. If you're not using Sphinx, you'll probably
# have a different build script.
cd doc
make clean html EXAMPLES_PATTERN=ex_*
cd ..

# Move the docs to the top-level directory, stash for checkout
mv doc/_build/html ./

# The html/ directory will stay there when we stash
git stash

# Checkout our gh-pages branch, remove everything but .git
git checkout gh-pages
git pull origin gh-pages

# Make sure to set the credentials! You'll need these environment vars
# set in the "Environment Variables" section in Circle CI
git config --global user.email "$GH_EMAIL" > /dev/null 2>&1
git config --global user.name "$GH_NAME" > /dev/null 2>&1

# Remove all files that are not in the .git dir
find . -not -name ".git/*" -type f -maxdepth 1 -delete

# Remove the remaining artifacts. Some of these are artifacts of the
# LAST gh-pages build, and others are remnants of the package itself.
# You will have to amend this to be more specific to your project.
declare -a leftover=(".cache/"
                     ".idea/"
                     "build/"
                     "build_tools/"
                     "doc/"
                     "examples/"
                     "gh_doc_automation/"
                     "gh_doc_automation.egg-info/"
                     "_downloads/"
                     "_images/"
                     "_modules/"
                     "_sources/"
                     "_static/"
                     "auto_examples/"
                     "includes"
                     "modules/")

# Check for each left over file/dir and remove it, or echo that it's
# not there.
for left in "${leftover[@]}"
do
    rm -r "$left" || echo "$left does not exist; will not remove"
done

# We need this empty file for git not to try to build a jekyll project.
# If your project *is* Jekyll, then this doesn't apply to you...
touch .nojekyll
mv html/* ./
rm -r html/

# Add everything, get ready for commit. But only do it if we're on
# master. If you want to deploy on different branches, you can change
# this.
if [[ "$CIRCLE_BRANCH" =~ ^master$|^[0-9]+\.[0-9]+\.X$ ]]; then
    git add --all
    git commit -m "[ci skip] publishing updated documentation..."

    # We have to re-add the origin with the GH_TOKEN credentials. You
    # will need this SSH key in your environment variables on Circle.
    # Make sure you change the <project>.git pattern at the end!
    git remote rm origin
    git remote add origin https://"$GH_NAME":"$GH_TOKEN"@github.com/"$GH_NAME"/gh_automation.git

    # NOW we should be able to push it
    git push origin gh-pages
else
    echo "Not on master, so won't push doc"
fi
```

I'm using Sphinx, so my documentation is built with the `doc/Makefile` that's created from the
[`sphinx-quickstart`](https://www.sphinx-doc.org/en/master/man/sphinx-quickstart.html). If you're using a different
framework, you'll have to make some corrections to this line:

```bash
cd doc
make clean html EXAMPLES_PATTERN=ex_*
cd ..
```

There are several points in the script where you'll have to make amendments with your
git project name to make sure you're pointing to the right place. You also need the script
to be executable before you even push, or you'll get a build failure!

```bash
$ chmod +x build_tools/circle/build_push_doc.sh
```

### Setting environment variables

In the script above, you'll notice that there are some environment variables you'll need to have stashed in the
"Environment Variables" section of Circle. Otherwise, you'll get an error like this:

![Circle failure](../img/failure-message.png)

There are three environment variables you'll need:

* `GH_NAME` - Your github username. Mine, for instance, is tgsmith61591
* `GH_EMAIL` - The email address you have associated with your github account.
* `GH_TOKEN` - This is an SSH key you're going to have to create on Github. Follow [these instructions](https://help.github.com/articles/creating-a-personal-access-token-for-the-command-line/)

All of these are going to be necessary for you to be able to push to your remote from
Circle, and the token in particular should be closely guarded.


### Amending your `config.yml`

At this point, you should have a Circle config file setup for your project, and you
should have your build script setup. Now we're going to amend your config with the
documentation build/push workflow.

All you need to do is to create a new section in your YAML file. Here's what mine looks
like in the [example project](https://github.com/tgsmith61591/gh_automation):

```yaml
deploy-doc:
  docker:
  - image: python:3.6
  working_directory: ~/gh_doc_automation
  steps:
  - checkout
  - run: make doc-requirements
  - run: make install
  - run: ./build_tools/circle/build_push_doc.sh
```

You can use a different Python image if you want, and you'll want to set the working directory to your project
name. Notice also that I'm building from a `Makefile`. All the `Makefile` does is alias away a series of commands
behind one "target". Here's what happens when `make doc-requirements` is called:

```bash
$ python -m pip install -r build_tools/doc/doc_requirements.txt
```

Using a `Makefile` is completely a taste thing and is by no means required for this to work. Without a `Makefile`, my config would look like:

```yaml
deploy-doc:
  docker:
  - image: python:3.6
  working_directory: ~/gh_doc_automation
  steps:
  - checkout
  - run: python -m pip install -r build_tools/doc/doc_requirements.txt
  - run: python -m pip install -r requirements.txt
  - run: python setup.py install
  - run: ./build_tools/circle/build_push_doc.sh
```

Once you have added the job to your config file, you need to make sure it runs. Add the job to your existing workflow:

```yaml
workflows:
  version: 2
  your_pipeline:
    jobs:
      - some-other-job
      - deploy-doc
```

And **that's all!** Next time you push to master, your documentation will be built and deployed! You can view it at:

```
http://<GH_NAME>.github.io/<YOUR_PROJECT_NAME>
```

### Final thoughts

This is a simple approach to automating your documentation pushes. Of course, there will be corner cases where this isn't exactly what
you're after, but it should provide you with what you need to adapt the solution to your requirements. You may also be interested
in checking out the way I build [versioned documentation](https://github.com/tgsmith61591/pmdarima/blob/develop/build_tools/circle/build_push_doc.sh) in my project, [`pmdarima`](https://github.com/tgsmith61591/pmdarima).
