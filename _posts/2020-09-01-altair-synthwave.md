---
title: Synthwave Styling Data Visualizations in Python with Altair
layout: post_norel
permalink: /pages/projects/synthwave-styling-with-altair
description: "The dream of the '80s is alive in my new favorite Python visualization tool."
---

![synthwave](/images/projects/altair-synthwave/synthwave.jpg)

_This was originally published [on the Pluralsight tech blog](https://www.pluralsight.com/tech-blog/synthwave-styling-data-visualizations-in-python-with-altair/)._

The [Synthwave](https://en.wikipedia.org/wiki/Synthwave) genre of electronic music brings back the aesthetic of 1980s film and music -- and its bright, retro-futuristic style has crossed over into games like _Grand Theft Auto: Vice City_ and _Far Cry 3: Blood Dragon_.
You can even bask in the neon glow of a [VS Code theme](https://marketplace.visualstudio.com/items?itemName=RobbOwen.synthwave-vscode) (and with the glow effect disabled, it's actually a really solid dark mode -- I'm writing this on it right now).
Dominik Haitz's [excellent post on the Matplotlib blog](https://matplotlib.org/matplotblog/posts/matplotlib-cyberpunk-style/) walks through how to apply the style to your visualizations in Python.
But what if we don't want to use Matplotlib?

While [Matplotlib](https://matplotlib.org/) is undeniably the dominant player for visualization in Python, I've always found it a bit jumbled.
Mixing and matching object-oriented and imperative syntax (via its `pyplot` submodule), or bouncing between Pythonic construction and its original MATLAB-like design, results in a rather unintuitive and difficult interface that makes it overly difficult to express what you actually _want_ to do with your visualizations.
While the [seaborn](https://seaborn.pydata.org/) wrapper makes this easier, deep customization still requires digging into Matplotlib internals.
In fact, the difficulty of data visualization in Python is one of the most frequent arguments I hear in favor of R as a data-science language.

More recently, a number of alternate visualization libraries have emerged -- [Bokeh](https://docs.bokeh.org/en/latest/), [Plotly](https://plotly.com/dash/), and [Altair](https://altair-viz.github.io/), to name a few, all start from scratch to design a more consistent & powerful visualization experience for Python.
In particular, I've recently started working with Altair, a Python wrapper around the [Vega](https://vega.github.io/vega/) (or more properly, [Vega-Lite](https://vega.github.io/vega-lite/)) declarative visualization grammar.
This means that, in Altair, I simply need to describe _what_ I want the plot to do, in contrast to _how_ to do it (which in Matplotlib generally means jumping through some hidden-state hoops), resulting in a rich, interactive visualization that can be exported to live Javascript, HTML, or static images.

So, naturally, I want to bring that same Synthwave styling to Altair visualization!
In this post, we'll walk through creating a simple visualization, and customizing it both on-the-fly in the plot itself and by laying out a reusable _Altair theme_.

Let's dive in!

## first steps

Since Altair produces Vega visualizations, we'll need a Javascript frontend to display our charts.
Fortunately, Altair's [renderers](https://altair-viz.github.io/user_guide/display_frontends.html) work out of the box with Jupyter notebooks, so I'd recommend working that way for prototyping your visualizations.
If you're not working in a Jupyter notebook, installing the `altair_viewer` package will let you spawn and view visualizations from your terminal -- or, you can just call `.save("html")` on your chart object to produce a file that can be viewed in any web browser.

First, we'll need some sample data -- let's generate a pair of Gaussians that we can easily visualize, as well as a categorical label for them.

```python
x = numpy.arange(-4, 4, 0.1)
y_left = scipy.stats.norm.pdf(x, loc=-1)
y_right = scipy.stats.norm.pdf(x, loc=1)
df = pandas.concat([
    pandas.DataFrame({"x": x, "y": y_left, "z": "left"}),
    pandas.DataFrame({"x": x, "y": y_right, "z": "right"})
]).sort_values(by="x")
```

Altair visualizations are built around the `Chart` object -- this keeps track of both our data (in the form of a `pandas.DataFrame`) and the state of the visualization.
Altair is designed to be fully declarative: that is, every specification for the plot is declared as an operation on the `Chart` object, generally just describing what we want to alter (leaving the _how_ to Vega-Lite's internals).
Each operation returns an updated version of the `Chart`, so operations can be daisy-chained together to describe the plot.
To begin:

```python
altair.Chart(df).encode(x="x", y="y", color="z").mark_line()
```

Here, we have:

1. created our `Chart` object (with included data)
2. _encoded_ data (that is, we've tied specific data columns to visual "channels", like the x- and y-axes and color level)
3. declared a `line` mark with our data to display

We're already off to a good start -- with this minimal code, we've already produced a decent-looking plot, simply by describing to Altair what we wanted on it.

![01](/images/projects/altair-synthwave/01-base.svg)

We can add more detail to this at the `Chart` level by providing annotations in the encoding:

```python
chart = (
    altair.Chart(df)
    .encode(
        x=altair.X("x", title="x-range"),
        y=altair.Y("y", title="gaussian pdf"),
        color=altair.Color("z:N", title="distribution")
    )
)
chart.mark_line()
```

where the altair schema objects (`X`, `Y`, `Color`) let us manipulate the encoding.
We can also use a shorthand directly in the data fields: for example, the annotation `z:N` indicates that the column should be treated as categorical.
While Altair is mostly clever about this, it can be helpful to indicate column types, as Altair will intelligently handle categorical, ordinal, and timestamp fields as well as the default continuous values -- we can apply [aggregations and transforms](https://altair-viz.github.io/user_guide/transform/aggregate.html) in this way as well!

![02](/images/projects/altair-synthwave/02-base-annotated.svg)

I feel pretty good about this -- the chart is expressing what we want it to express.
Now, let's turn to styling it!

## color palettes and themes

Altair allows chart configuration at any level of granularity: global defaults and themes, chart-level preferences, or local overrides applied to specific marks.
We could apply our styling to a single chart via the [`Chart.configure_*` methods](https://altair-viz.github.io/user_guide/configuration.html), but if we want to reuse a consistent styling for our visualizations, we should encode our changes in an _Altair theme_.

Since Vega expresses visualizations in a JSON-serializable form, to design a theme we just need to design a function returning a dictionary containing our desired settings (overriding the global defaults), and register that as a theme.
For example, to set the default chart size:

```python
def synthwave():
    return {
        "config": {
            "view": {
                "continuousWidth": 400,
                "continuousHeight": 300
            }
        }
    }

altair.themes.register("synthwave", synthwave)
altair.themes.enable("synthwave")
```

(Note that we've specifically set sizes for continuous-value axes -- we can set defaults separately for different data encodings).
We can start setting color palettes by applying the necessary values in this config function.

Let's start with the plotting area.
The background color is a straightforward top-level configuration, while the plotting details of the axes are a bit more intricate.
We can separately set configuration for the grid (lines on the plot itself), domain (available space for data channels -- in this case, sidebars for the x- and y-axes), and ticks, overriding their colors or nulling them out.

```python
def synthwave():
    background = "#2e2157"  # dark blue-grey
    grid = "#2a3459"        # lighter blue-grey

    return {
        "config": {
            "view": {
                "continuousWidth": 400,
                "continuousHeight": 300
            },
            "background": background,
            "axis": {
                "gridColor": grid,
                "domainColor": None,
                "tickColor": None
            }
        }
    }
```

![03](/images/projects/altair-synthwave/03-background.svg)

Of course, now we can't read our labels, so we need to override those as well.
Again, the configuration is quite granular, so (for example) we can separately style the tick labels and axis titles under the `axis` configuration, as well as add a config block for the `legend`:

```python
def synthwave():
    background = "#2e2157"  # dark blue-grey
    grid = "#2a3459"        # lighter blue-grey
    text = "#d3d3d3"        # light grey

    return {
        "config": {
            "view": {
                "continuousWidth": 400,
                "continuousHeight": 300
            },
            "background": background,
            "axis": {
                "gridColor": grid,
                "domainColor": None,
                "tickColor": None,
                "labelColor": text,
                "titleColor": text
            },
            "legend": {
                "labelColor": text,
                "titleColor": text
            }
        }
    }
```

![05](/images/projects/altair-synthwave/05-legend.svg)

Alright!
Now that we've got our plotting surface styled, we're ready to work on our data.
Much like how Altair encodes the potential inputs for a data channel as the _domain_, the representation of its output is encoded in the _range_.
This lets us separately configure how the plot will represent categorical versus ordinal versus continuous data.
For our two distributions, we just need to configure a categorical range with a discrete color palette to draw from:

```python
def synthwave():
    background = "#2e2157"  # dark blue-grey
    grid = "#2a3459"        # lighter blue-grey
    text = "#d3d3d3"        # light grey
    line_colors = [
        "#2de2e6",          # teal/cyan
        "#fe53bb",          # pink
        "#f5d300",          # yellow
        "#00ff41",          # matrix green
        "#ff6c11",          # hot orange
        "#fd1d53"           # hot red
    ]

    return {
        "config": {
            "view": {
                "continuousWidth": 400,
                "continuousHeight": 300
            },
            "background": background,
            "axis": {
                "gridColor": grid,
                "domainColor": None,
                "tickColor": None,
                "labelColor": text,
                "titleColor": text
            },
            "legend": {
                "labelColor": text,
                "titleColor": text
            },
            "range": {
                "category": line_colors
            }
        }
    }
```

![06](/images/projects/altair-synthwave/06-linecolor.svg)

## curve styling

Next, we want to add a fill effect under our curves, with just a hint of the color marking the curve itself.
This means we're actually altering our mark -- instead of a line, we'll be marking off an _area_.
Altair's `mark_area` also lets us add a boundary line, so we can retain the original look and style both:

```python
def synthwave():
    background = "#2e2157"  # dark blue-grey
    grid = "#2a3459"        # lighter blue-grey
    text = "#d3d3d3"        # light grey
    line_colors = [
        "#2de2e6",          # teal/cyan
        "#fe53bb",          # pink
        "#f5d300",          # yellow
        "#00ff41",          # matrix green
        "#ff6c11",          # hot orange
        "#fd1d53"           # hot red
    ]

    return {
        "config": {
            "view": {
                "continuousWidth": 400,
                "continuousHeight": 300
            },
            "background": background,
            "axis": {
                "gridColor": grid,
                "domainColor": None,
                "tickColor": None,
                "labelColor": text,
                "titleColor": text
            },
            "legend": {
                "labelColor": text,
                "titleColor": text
            },
            "range": {
                "category": line_colors
            },
            "area": {
                "line": True,
                "fillOpacity": 0.1
            },
            "line": {
                "strokeWidth": 2
            }
        }
    }
```

Here we've marked the area to fill with 10% opacity by default, and to draw its bounding line -- this will use the `line` configuration just like a standalone `line` mark would.
To display, we simply replace our call to `mark_line` with `chart.mark_area()`, resulting in

![07](/images/projects/altair-synthwave/07-fill.svg)

## make it glow

Of course, the crowning glory of any Synthwave effect is the blurred glow emanating from our sharp, bright lines.
We can achieve this effect pretty easily by mirroring the Matplotlib approach, redrawing successively wider lines at low opacity to achieve a denser effect the closer to the main line we get.

This can be a little nonintuitive to do in Altair's declarative syntax -- i.e., we can't just replot things in a for loop like in `matplotlib.pyplot`, since each mark overrides the previous.
However, Altair does provide prinicipled ways to merge plotted objects into a single compound chart, with vertical or horizontal concatenation, repeats & facets, or layered charts.
In the `LayeredChart`, we can provide a variadic set of input  objects, and collate them into a single chart object overlaid on shared axes.
By successively drawing our lines with `mark_line`, we can then layer them into a single chart object that will behave just like one produced by any of the built-in `mark_*` functions.

```python
def mark_blurred_line(chart, n_glows=10, base_opacity=0.3):
    opacity = base_opacity/n_glows
    glows = (
        chart.mark_line(opacity=opacity, strokeWidth = 2 + (1.05 * i))
        for i in range(1, n_glows + 1)
    )
    return altair.layer(*glows)

# using our chart object from before
fill = chart.mark_area()
blur = mark_blurred_line(chart)
altair.layer(fill, blur)
```

![08](/images/projects/altair-synthwave/08-glow.svg)

## wrapping up

And that's it!
With relatively little code we've achieved some really striking visualizations, and we can reuse it for other visualizations just by enabling the theme.
Of course, there's more to be done (like properly styling the color palette to handle ordinal or continuous ranges), but that's just a question of [populating more configuration options](https://vega.github.io/vega-lite/docs/config.html).
I suspect there's a cleverer way to create the glow effect, as that should be doable with a CSS drop-shadow, but I haven't yet worked out integrating that.

We're not restricted to making static graphics either -- since Altair builds fully-spec'd Vega graphics, we can export these directly as HTML, or as a JSON schema that can be displayed interactively by a JavaScript front-end.
While I suspect I'll still be doing some quick-and-dirty plotting in Matplotlib (usually via seaborn), I've been really pleased with how _easy_ it is to make compelling visualizations in Altair, without feeling like I'm fighting against my visualization tool.

And yes, I was [listening to this](https://open.spotify.com/playlist/3gWAZPuNWpELIhKNbnpfwk?si=2vBWcqUeSTOcvF2XQR6M5A) while writing.
