---
title: I-Mode Energy Confinement Modeling (doctoral thesis)
layout: post
permalink: /pages/projects/imode-confinement
---

The I-mode is a promising new operating regime for tokamak fusion reactor designs. As a component of my thesis research, I developed the first comprehensive study of the [pedestal structure in I-mode](/pages/projects/imode-pedestal), using a series of dedicated experiments on MIT’s Alcator C-Mod tokamak.  In conjunction with this, my thesis work included an examination of the impact of the pedestal on global energy confinement in I-mode - the understanding of which is essential for extrapolating the regime to reactor-scale tokamaks.

The I-mode' strong temperature pedestal supports very high core temperatures relative to comparable H-modes, such that at moderate densities the I-mode can attain competitive core pressure (thus fusion power density) despite the relaxed, [naturally-stable](/pages/projects/imode-stability) pedestal.  

However, due to the markedly different physics governing the temperature pedestal and energy confinement in I-mode, previously-established empirical trends for energy confinement time in L-mode and H-mode (termed the ITER89 and ITER98 scalings, respectively) must be supplemented with a new scaling for I-mode confinement.  These empirical scalings take the form of a power law based on a number of engineering parameters, in the form

![scalinglaw](/images/projects/imode-confinement/scalinglaw.jpg){: .inline-img}

for plasma current $$I_p$$, applied magnetic field $$B_T$$, average density $$\overline{n}_e$$, machine major radius and aspect ratio $$R$$ and $$\varepsilon$$, plasma elongation $$\kappa$$, and heating power $$P_{loss}$$ (which also accounts for changes in the thermal stored energy in the plasma).

To conduct this analysis of I-mode data, I used a straightforward [linear regression analysis](https://github.com/jrwalk/confinement) of the logarithm of the power law (allowing for a wide range in the fitting parameters to be included), built on a flexible script producing a power-law model for an arbitrary set of input parameters.  For a first pass, I began with the included parameters from the ITER98 H-mode scaling.  However, the machine size and plasma shaping parameters must immediately be discarded, as the input parameters from a single tokamak experiment vary far too little to produce a meaningful result (the ITER89 and ITER98 scalings were conducted on an extensive multi-machine database - I am currently assembling I-mode data from other devices to extend my own confinement database).  Dropping these parameters results in an "irreducible-complexity" power-law fit suitable for single-machine use on C-Mod data, shown below as (b):

![table1](/images/projects/imode-confinement/table1.jpg){: .inline-img}

The results from this fit are shown below, for the merged databases including older data from both magnetic-geometry configurations("rev-B" and "for-B").

![tauE_1](/images/projects/imode-confinement/tauE_1.jpg){: .inline-img}

These parameter exponents are shown compared to the results from the ITER89 and ITER98 scalings below.  The I-mode scaling is notable for its substantially stronger magnetic-field dependence, and much weaker degradation of energy confinement with increased heating power.

![table2](/images/projects/imode-confinement/table2.jpg){: .inline-img}

This is consistent with observed physics properties in I-mode - the weak degradation of $$\tau_E$$ with heating power is indicated by the strong response of the temperature pedestal (which is, fundamentally, representative of a barrier to energy loss).  The strong dependence on magnetic field may be tied to the effect of increased field on transition thresholds between L-, I-, and H-mode - increased magnetic field tends to suppress the transition to conventional H-mode (which is ordinarily the upper bound of the operating range in I-mode), thus allowing the operator to push the plasma more aggressively without triggering the H-mode.

The implications of this new scaling for larger tokamaks may be examined by making an estimate of the machine-size dependence in the scaling (which we omitted due to the negligible variation in input parameter values).  Initial indications using I-mode data from the ASDEX Upgrade (AUG) tokamak in Germany suggests a size dependence similar to that in the H-mode (ITER98) scaling.  Using this estimate, I-mode confinement levels may be extrapolated to other major tokamaks:

![tauE_2](/images/projects/imode-confinement/tauE_2.jpg){: .inline-img}

The scaling reasonably captures the range of I-mode confinement data from ASDEX Upgrade, while also capturing the comparatively poor confinement initially observed in I-mode experiments on the DIII-D tokamak in San Diego (the $$H_{98} = 1$$ line indicates the H-mode confinement prediction for identical input parameters).  However, when extended to large, high-power, higher-field devices (JET in the UK, and the ITER tokamak currently under construction in France), the strong field dependence and weak degradation of $$\tau_E$$ with heating power dominates the scaling, resulting in predicted energy confinement well in excess of H-mode levels - an exciting result suggesting that I-mode operation could lead to more compact, economical, high-field/high-power tokamak fusion power plants.