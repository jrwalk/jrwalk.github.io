---
title: The Pedestal
layout: post
permalink: pedestal
---

<p>To reach sufficient fusion power gain, a tokamak needs to reach a minimum value of thermal pressure and energy confinement time -- the former sets the fusion power density, and the latter measures how well the plasma retains its heat (thus how well heat from fusion can maintain the necessary conditions for a self-sustaining reaction).
The maximum pressure (relative to magnetic field strength) is set by stability limits: we need to make up the difference with confinement improvements.</p>

<p>By default, both energy and particles are driven rapidly from the plasma core by turbulence -- and this transport gets worse with more heating power!
The brute force approach is to run in a larger machine with stronger magnetic fields to compensate, but this is expensive.  <strong>In its default state, this plasma is not suitable for an economical power plant.</strong></p>

<p>Under the right conditions (shaping, heating power) the plasma transitions to "high-confinement" or H-mode (compared to the default "low-confinement" L-mode).
The plasma forms a transport barrier, the <strong>pedestal</strong>, in the edge, where sheared flows cut off turbulent transport -- density and temperature profiles ``pile up'' behind this barrier, leading to much higher core pressure and fusion power density.</p>

![pedestal](/images/pedestal/prof_L_H.jpg){: .inline-img}

![elmfree](/images/pedestal/trace_1110114003_nofluct.jpg){: .wrap-img}
<p>This presents problems, however: the plasma retains impurities (metal wall material, helium "fusion ash") as well as fuel ions, leading to radiative losses that eventually overcome the heating power, dropping the plasma back into L-mode.  <strong>This H-mode is an inherently transient state.</strong></p>

<p>So what we need is high <i>energy</i> confinement specifically, with low <i>particle</i> confinement (low enough, at least, to avoid impurity buildup).
And that's it, right?</p>

<p>Under the right conditions, the plasma develops <strong>Edge-Localized Modes</strong> (ELMs) -- periodic instabilities that relax the pedestal, driving a burst of energy and particles out of the plasma.
This is sufficient to prevent impurity accumulation.</p>
![elmy](/images/pedestal/trace_1101214029.jpg){: .wrap-img}
<p>On current experiments, this is the baseline for high performance, and was thought to be acceptable for fusion reactor use.  However, on a reactor-scale device, these ELMs drive enormous pulsed heat loads -- effectively like detonating a hand grenade in the exhaust several times per second, far in excess of material tolerances for the reactor wall.</p>

<p>So what we need, then, is high energy confinement, low particle confinement, and to avoid (or mitigate or suppress) large ELMs.
Fortunately, a number of solutions exist: engineering fixes (external controls that suppress or smooth out the ELM cycle), and physics solutions in the form of steady H-modes without ELMs, where the pedestal is regulated by some benign fluctuation below the ELM limit.  A number of these regimes exist, but each has its pros and cons -- high-performance operation without ELMs, then, is a major research focus.</p>