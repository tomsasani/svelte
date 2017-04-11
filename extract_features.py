import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cyvcf2 import VCF
import math
import pysam
import seaborn as sns
from mods import count_support, delta_depth, flank_is_ref
import argparse
import numpy as np

sns.set(style='ticks', palette='Set2')

p = argparse.ArgumentParser()
p.add_argument("--bam", required=True, help="BAM with all alignments")
p.add_argument("--vcf", required=True, help="VCF containing SVs you wish to genotype")
p.add_argument("-tech", help="sequencing technology", default='ont')
args = p.parse_args()

num_entries = sum([1 for v in VCF(args.vcf)])

bam = pysam.Samfile(args.bam, "rb")
vcf = VCF(args.vcf)

# Open output feature and label files.
features_out = open('features.npy', 'w')
labels_out = open('labels.npy', 'w')

smp2idx = dict(zip(vcf.samples, range(len(vcf.samples))))

color_scheme = dict(zip([0, 1, 3], ['green', 'red', 'blue']))
gt_names = dict(zip([0, 1, 3], ['homozygous reference', 'heterozygous', 'homozygous alternate']))
gt_to_bin = dict(zip([0, 1, 3], [-1, 0, 1]))

# Initialize the feature and label arrays.
X = np.ones([1,5], int) 
y = np.zeros([1,], int) 

window = 20
counter = 0
for v in vcf:
    left, right = v.start, v.INFO["END"]
    mid = left + (right - left) / 2

    # If no SVLEN is reported, we'll calculate it ourselves.
    try:
        svlen = v.INFO["SVLEN"]
    except KeyError:
        svlen = int(v.INFO["END"] - v.POS)
    svtype = v.INFO["SVTYPE"]
    num_samples = len(v.gt_types)

    # Grab the "gold standard" genotype corresponding to 
    # NA12878, and add it to the labels array.
    NA12878_idx = smp2idx['NA12878']
    gt = v.gt_types[NA12878_idx]
    training_gt = np.array([gt_to_bin[gt]])
    y = np.append(y, training_gt, axis=0)

    ls, le = left - window, left + window
    rs, re = right - window, right + window

    combined_vector = []
    plot_vals = []

    # Count ALT and REF read support at both breakpoints.
    for (end, s, e, pos) in [("left", ls, le, left), ("right", rs, re, right)]:

        ref_reads, alt_reads = count_support(bam, v.CHROM, s, e, end, tech=args.tech)

        # If using ONT reads, implement a separate function to 
        # determine whether the breakpoints are "noisy."
        if args.tech == 'ont':
            if flank_is_ref(bam, v.CHROM, s, e):
                ref_reads += (ref_reads / 3)
            else: alt_reads +=  (alt_reads / 3)

        # Convert counts of REF and ALT support to proportions
        # of total read support. Improves performance.
        r_sum = ref_reads + alt_reads
        if r_sum == 0:
            ref, alt = 0, 1
        else:
            ref = ref_reads / float(r_sum)
            alt = alt_reads / float(r_sum)

        # Add REF and ALT support proportions to the "combined_vector."
        combined_vector.extend([ref])
        combined_vector.extend([alt])

        # Plotting our training data for manuscript.
        plot_vals.extend([ref_reads])
        plot_vals.extend([alt_reads])
    
    # To reduce dimensions, plot will be combined REF support vs. combined ALT.
    x_ = plot_vals[0] + plot_vals[2]
    y_ = plot_vals[1] + plot_vals[3]

    # Plot features for first 500 values.
    if counter < 750:
        if (x_ < 200 and y_ < 200):
            plt.scatter(x_, y_, edgecolors=color_scheme[int(gt)], alpha=1.0, facecolors='none')

    combined_vector.extend([round(delta_depth(bam, v.CHROM, left, right), 2)])
    if math.isnan(float(combined_vector[-1])):
        combined_vector[-1] = 1.0

    combined_vector = np.array(combined_vector)
    X = np.append(X, np.array([combined_vector]), axis=0)
    print "done with variant %s of %s" % (counter + 1, num_entries)
    counter += 1

# Save our features and labels as a numpy file.
np.save(features_out, X)
np.save(labels_out, y)

sns.despine(right=True, top=True)
plt.xlabel("Total 'reference' support")
plt.ylabel("Total 'alternate' support")
plt.savefig('spread.png', dpi=300)
