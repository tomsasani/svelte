import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from cyvcf2 import VCF
import math
import pysam
from mods import count_support, delta_depth, flank_is_ref
from operator import itemgetter
import argparse
import random
import toolshed as ts
from tempfile import TemporaryFile
import sys
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--bam", required=True, help="BAM with all alignments")
p.add_argument("--vcf", required=True, help="VCF containing SVs you wish to genotype")
p.add_argument("-training", help="extract test or training data", action="store_true")
p.add_argument("-tech", help="sequencing technology", default='ont')
args = p.parse_args()

bam = pysam.Samfile(args.bam, "rb")

if args.training:
    features_out = open('1KG.features.npy', 'w')
    labels_out = open('1KG.labels.npy', 'w')
else:
    features_out = open('test.features.npy', 'w')
    labels_out = open('test.labels.npy', 'w')

counter = 0
vcf = VCF(args.vcf)
smp2idx = dict(zip(vcf.samples, range(len(vcf.samples))))

color_scheme = ['blue', 'red', 'filler', 'green']
gt_names = ['homozygous reference', 'heterozygous', 'filler', 'homozygous alternate']
X = np.ones([1,5], int) 
window = 20
y = np.zeros([1,], int) 
for v in vcf:
    if args.training:
        if v.CHROM != '20':
            continue
    else:
        if counter > 200:
            continue
    left = v.start
    right = v.INFO["END"]
    mid = left + (right - left) / 2
    try:
        ci = v.INFO["CIPOS"]
        ce = v.INFO["CIEND"]
    except KeyError:
        continue
    try:
        svlen = v.INFO["SVLEN"]
    except KeyError:
        svlen = int(v.INFO["END"] - v.POS)
    svtype = v.INFO["SVTYPE"]
    num_samples = len(v.gt_types)
    NA12878_idx = smp2idx['NA12878']
    training_gt = np.array([v.gt_types[NA12878_idx]])
    y = np.append(y, training_gt, axis=0)

    ls, le = left - window, left + window
    rs, re = right - window, right + window
    ms, me = mid - window, mid + window

    ends = [left, right]

    combined_vector = []

    for (end, s, e, pos) in [("left", ls, le, left), ("right", rs, re, right)]:

        ref_reads, alt_reads = count_support(bam, v.CHROM, s, e, end, tech=args.tech)

        if args.tech == 'ont':
            if flank_is_ref(bam, v.CHROM, s, e):
                ref_reads += (ref_reads / 3)
            else: alt_reads +=  (alt_reads / 3)

        r_sum = ref_reads + alt_reads
        if r_sum == 0:
            ref, alt = 0, 1
        else:
            ref = ref_reads / float(r_sum)
            alt = alt_reads / float(r_sum)
        combined_vector.extend([ref])
        combined_vector.extend([alt])
    x_ = combined_vector[0] + combined_vector[2]
    y_ = combined_vector[1] + combined_vector[3]

    if x_ > 300 or y_ > 300:
        continue
    if args.training:
        plt.scatter(x_, y_, color=color_scheme[int(training_gt)], marker='+', label=gt_names[int(training_gt)])

    combined_vector.extend([round(delta_depth(bam, v.CHROM, left, right), 2)])
    if math.isnan(float(combined_vector[-1])):
        combined_vector[-1] = 1.0

    combined_vector = np.array(combined_vector)
    X = np.append(X, np.array([combined_vector]), axis=0)
    counter += 1
    print "Done with variant #%s" % (counter)

np.save(features_out, X)
np.save(labels_out, y)
plt.xlabel("Total 'reference' support")
plt.ylabel("Total 'alternate' support")
plt.savefig('spread.png', dpi=300)
