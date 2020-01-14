---
title: Managing Python Environments
layout: post_norel
permalink: /pages/projects/managing-python-environments
---

tl;dr [read more here](https://www.pluralsight.com/tech-blog/managing-python-environments/)

It's really, really easy for your Python environment(s) to become a [messy, convoluted disaster](https://xkcd.com/1987/).
What's worse, there's a whole zoo of tools that can interact with each other in weird ways (or not at all) for managing Python environments - this can be one of the biggest hurdles a new Python developer will face.
That said, most any solution is workable, provided we agree on a few principles:

- **virtualization is your friend:** isolating your Python environment per-project makes your life infinitely easier by avoiding dependency clashes between projects
- **projects should be reproducible:** the more tightly you can specify dependencies, the easier it is to exactly reproduce the running environment for your code on the fly, for yourself or another dev
- **self-contained = deployable:** the easier it is to pack up and ship an environment with all the trimmings, the easier it is to get projects running on radically different systems (like moving from a dev environment to deployment)

I recently published an article through Pluralsight's tech blog on managing Python environments, available [in this post](https://www.pluralsight.com/tech-blog/managing-python-environments/).
I begin with the simplest low-level standard tooling, and work our way through newer, more powerful (though sometimes more esoteric and restrictive) options, discussing pros and cons along the way - I encourage you to consider all the options that could suit your project’s needs.
Enjoy!
