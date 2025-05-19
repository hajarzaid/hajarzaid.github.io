---
title: "Which Neurons Drive the Latent Space?"
parent: Blogs
layout: home
nav_order: 1
---

<!-- BEGIN: MathJax -->
<div>
<script>
  window.MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['\\(', '\\)']],
      displayMath: [['$$', '$$'], ['\\[', '\\]']]
    }
  };
</script>
<script id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>
</div>
<!-- END: MathJax -->

This short article is inspired by my experience reading computational
neuroscience papers. A common strategy for making neural models more
interpretable involves projecting high-dimensional neural activity into
a lower-dimensional space. This makes intuitive sense. It is difficult
to grasp the relationships between hundreds or thousands of neurons in
their original high-dimensional setting.

A closer look at dimensionality reduction methods like PCA, ISOMAP, and
LDA reveals that these latent spaces are not entirely abstract. They are
defined by explicit bases, which we can access and analyze. In doing so,
we can begin to ask which neurons are responsible for shaping these
low-dimensional dynamics.

## Background

We begin with a data matrix: $$X \in \mathbb{R}^{N \times T}$$ where $N$
is the number of neurons and $T$ is the number of timepoints.

PCA proceeds by computing the singular value decomposition (SVD) of the
mean-centered matrix $X$: $$\tilde{X} = U S V^\top$$ where
$U \in \mathbb{R}^{N \times N}$ contains the left singular vectors
(neuron-space directions), $S$ is a diagonal matrix of singular values
(scaling each mode), and $V^\top \in \mathbb{R}^{T \times T}$ contains
the right singular vectors (time-space directions).

The columns of $U$ define weighted combinations of neurons that form
axes in the latent space. The columns of $V$ describe how each component
evolves over time, and the singular values in $S$ represent the amount
of variance explained by each component.

Each principal component can be written as:
$$PC_k(t) = \sigma_k \cdot u_k v_k^\top(t)$$ This expresses the rank-1
spatiotemporal pattern captured by the $k^\text{th}$ component. The
vector $u_k$ lies in neuron space, indicating the contribution weights
of each neuron. The vector $v_k^\top(t)$ describes the temporal
evolution of this pattern, and $\sigma_k$ reflects how important the
component is in terms of variance.

The first principal component represents the dominant joint activity
pattern, which is a specific combination of neurons that varies together
over time in a meaningful way. So while PCA summarizes population
activity in low dimensions, it does not explicitly tell us which neurons
contribute when. The entries $U_{ik}$ tell us how much neuron $i$
contributes to the $k^\text{th}$ principal axis, but this contribution
is fixed and does not evolve with time.

## Constructing a Time-Resolved Contribution Basis

Here I propose a simple but useful idea. We can back-project to trace
how individual neurons contribute to the expression of a latent
trajectory over time. Once we have projected our high-dimensional neural
activity into a lower-dimensional latent space using PCA, we do not have
to treat the latent variables as black boxes. Each principal component
is built from a specific combination of neurons, and we can measure how
much each neuron is driving that component at any given time.

Given the activity of a neuron $x_i(t)$, and its weight in the $k^{\text{th}}$ principal component $u_{ik}$, the contribution of neuron $i$ to the expression of PC $k$ at time $t$ is:


$$C_{ik}(t) = u_{ik} \cdot x_i(t)$$

This gives us a time-resolved measure of how much each neuron is
influencing a given latent trajectory. Since the full projection onto PC
$k$ is just the dot product:

$$z_k(t) = \sum_i u_{ik} \cdot x_i(t)$$

we can break that sum down into individual neuron contributions. If you
want to know which neurons are responsible for a shift in the latent
space at a specific moment, this gives you a direct way to trace it.

Optionally, you could normalize these contributions:

$$\text{Relative contribution}_{ik}(t) = \frac{u_{ik} \cdot x_i(t)}{z_k(t)}$$

to get a sense of how much each neuron accounts for the total projection
at each timepoint. Positive and negative contributions can cancel out,
so the sign matters.

This idea feels almost too simple, but I have not seen it presented
explicitly in the literature. Most papers that use PCA treat the
components as summary axes, but do not trace how the expression of those
axes arises from individual neurons over time. Doing so can tell us a
lot about how neural roles shift during different behavioral epochs or
internal states, and potentially how information flows across a circuit.

## Why This Matters for Dynamical Systems 

This idea of tracing neuron contributions over time becomes especially
important when thinking about neural activity as a dynamical system. In
that framing, we are not just looking at neural activity as a static
cloud of points. We are watching how population activity moves through
state space. We care about trajectories, flows, and the directions the
system bends into over time.

At one point, I tried to dive into this directly using real neural data
from the Allen Brain Observatory. I was interested in how signals might
propagate through a population, especially during natural movie stimuli.
The idea was to iteratively estimate the Jacobian, the local
linearization of the system, and use it to infer how activity at time
$t$ influenced activity at time $t+1$. But I quickly ran into a wall.
The Jacobians were huge, noisy, and nearly impossible to interpret. Even
when I applied dimensionality reduction, I could not figure out how to
relate the compressed trajectories back to specific neurons. I ended up
with abstract dynamics that I could not pin to anything biologically
meaningful.

That failure made me step back and rethink the entire pipeline. I
realized I was trying to read the system's behavior before I had
identified its parts. That led me to this much simpler idea: if we start
with the latent space, can we at least say who is responsible for what?
Can we trace a latent trajectory back to the neurons driving it, and see
how those roles change as time unfolds?

## Toward Understanding Attractors in Neural Systems 

One possible application of this approach is in studying attractor
dynamics, the idea that neural populations may converge toward stable
activity patterns over time, depending on task, stimulus, or internal
state. Attractors are usually studied in latent space or inferred from
global population patterns, but what if we could instead track them
through the evolving roles of individual neurons?

If we compute the contribution matrix $C(t)$ , where each entry is:

$$C_{ik}(t) = u_{ik} \cdot x_i(t)$$

then we can define a time-varying contribution vector for each neuron:

$$\vec{c}_i(t) = [C_{i1}(t), C_{i2}(t), \dots, C_{iK}(t)]$$

This gives us a $K$-dimensional trajectory that describes how neuron
$i$'s influence on the latent space evolves over time. Doing this for
all neurons produces a structured view of how the population reorganizes
itself to drive state transitions.

From here, we can define a pairwise influence function between neurons:

$$A_{ij}(t) = \text{sim}(\vec{c}_i(t - \tau), \vec{c}_j(t))$$

where $\text{sim}(\cdot)$ could be a correlation, dot product, or
another causality metric, and $\tau$ is a time lag. This function can be
visualized as a dynamic influence heatmap, where the activation of one
neuron in latent space appears to modulate the future expression of
another.

By examining these influence maps over time, it may be possible to
detect repeating patterns, convergent configurations, or cyclical loops.
These are all hallmarks of attractor dynamics. Unlike traditional
phase-space analyses, this framework preserves neuron identity, which
means we can now ask: which neurons initiate transitions into attractor
basins? Which ones stabilize them? And how does this vary across trials
or behavioral conditions?

This kind of neuron-level tracing could offer a new lens on what it
means for a brain region to stabilize, shift, or reverberate \...and it
all starts from a simple back-projection.
