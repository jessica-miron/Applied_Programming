# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 09:01:07 2025

@author: jcmir
"""

import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


# Part 1
norm_samples = stats.norm.rvs(loc=0,scale=1,size=1000)
norm_samples_mean = np.mean(norm_samples)
norm_samples_std = np.std(norm_samples)
print(f"Normal Distribution Mean: {norm_samples_mean}\n")
print(f"Normal Distribution Std: {norm_samples_std}\n")

with PdfPages('my_multipage_plots.pdf') as pdf:
    fig, ax = plt.subplots(1,1)
    x = np.linspace(stats.norm.ppf(0.01),stats.norm.ppf(0.99),100)
    ax.plot(x, stats.norm.pdf(x),label="PDF")
    ax.hist(norm_samples,density=True, bins='auto',histtype='stepfilled',label="Random Samples")
    ax.set_xlim(x[0],x[-1])
    ax.set_xlabel("X")
    ax.set_ylabel("Magnitude")
    ax.set_title("Part 1: Normal Distribution")
    ax.legend()
    pdf.attach_note("Part 1 Plot")
    pdf.savefig()
    plt.close()

# Part 2
    binom_samples = stats.binom.rvs(n=10,p=0.5,size=1000)
    poisson_samples = stats.poisson.rvs(mu=5, size=1000)
    fig, [ax1,ax2] = plt.subplots(1,2,layout='constrained')
    ax1.hist(binom_samples,density=True,bins='auto',histtype='stepfilled',label='Binomial Distribution')
    ax2.hist(poisson_samples,density=True,bins='auto',histtype='stepfilled',label='Poisson Distribution')
    fig.suptitle("Part 2: Binomial vs Poisson")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Magnitude")
    ax1.set_title("Binomial Distribution")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Magnitude")
    ax2.set_title("Poisson Distribution")
    pdf.attach_note("Part 2 Plot")
    pdf.savefig()
    plt.close()

# Part 3
    uniform_array = []
    for i in range(30):
        uniform_samples = stats.uniform.rvs(size=100)
        mean_uniform_sample = np.mean(uniform_samples)
        uniform_array.append(mean_uniform_sample)
    fig, ax = plt.subplots(1,1)
    ax.hist(uniform_array,density=True,bins='auto',histtype='stepfilled',label='Sample means')
    x = np.linspace(stats.norm.ppf(0.01,loc=np.mean(uniform_array),scale=np.std(uniform_array)),stats.norm.ppf(0.99,loc=np.mean(uniform_array),scale=np.std(uniform_array)),100)
    ax.plot(x, stats.norm.pdf(x,loc=np.mean(uniform_array),scale=np.std(uniform_array)),label='Normal PDF')
    ax.set_xlabel('X')
    ax.set_ylabel('Magnitude')
    ax.set_title('Part 3: Uniform Samples Means Versus Normal Distribution')
    ax.legend()
    pdf.attach_note("Part 3 Plot")
    pdf.savefig()
    plt.close()


# Part 4
    expon_samples = stats.expon.rvs(loc=1/2,size=1000)
    fig, ax = plt.subplots(1,1)
    x = np.linspace(stats.expon.ppf(0.01),stats.expon.ppf(0.99),100)
    ax.plot(x, stats.expon.pdf(x),label="PDF")
    ax.hist(expon_samples,density=True, bins='auto',histtype='stepfilled',label="Random Samples")
    ax.set_xlim(x[0],x[-1])
    ax.set_xlabel("X")
    ax.set_ylabel("Magnitude")
    ax.set_title("Part 4: Exponential Distribution")
    ax.legend()
    pdf.attach_note("Part 4 Plot")
    pdf.savefig()
    plt.close()

# Part 5
    less_than_120 = stats.norm.cdf(120,loc=100,scale=15)
    percentile_95 = stats.norm.ppf(0.95,loc=100,scale=15)
    print(f"Probability Under 120: {less_than_120}\n")
    print(f"95th Percentile is: {percentile_95}\n")
    norm_samples_5 = stats.norm.rvs(loc=100, scale=15, size=10000)
    fig,ax = plt.subplots(1,1)
    ax.hist(norm_samples_5,density=True,bins='auto',histtype='stepfilled',label='Random Samples')
    ax.axvline(120,label='120',c='k')
    ax.axvline(percentile_95,label='95th Percentile',c='r')
    ax.set_xlabel("X")
    ax.set_ylabel("Magnitude")
    ax.set_title("Part 5: Normal Distibution (mean=100, std=15)")
    ax.legend()
    pdf.attach_note("Part 5 Plot")
    pdf.savefig()
    plt.close()

# Part 6
    data = [118, 125, 130, 110, 135, 142, 128, 120, 138, 145, 132, 126, 140, 150, 122, 134, 129, 136, 144, 127, 131, 119, 133, 141, 137, 124, 121, 139, 147, 143]
    loc1, scale1 = stats.norm.fit(data)
    over_140 = stats.norm.sf(140, loc1, scale1)
    print(f"Probability Over 140: {over_140}")
    print(f"loc:{loc1},scale:{scale1}")
    fig,ax=plt.subplots(1,1)
    ax.hist(data, density=True,bins='auto',histtype='stepfilled',label='Measurements')
    x=np.linspace(stats.norm.ppf(0.01,loc=loc1,scale=scale1),stats.norm.ppf(0.99,loc=loc1,scale=scale1),100)
    ax.plot(x, stats.norm.pdf(x,loc=loc1,scale=scale1),label='Normal Fit',c='k')
    ax.axvline(140,label='140',c='r')
    ax.set_xlabel("Blood Pressure (mmHg)")
    ax.set_ylabel("Magnitude")
    ax.set_title("Part 6: Blood Pressure Distribution")
    pdf.attach_note("Part 6 Plot")
    pdf.savefig()
    plt.close()