bam = "/scratch/ucgd/lustre/u1006375/sv_sensitivity_ont/nanopore-NA12878/nanopore-NA12878.mmp.merged.bam"
ref = "/scratch/ucgd/lustre/work/u6000771/bcbio/genomes/Hsapiens/g1k_v37_decoy/seq/g1k_v37_decoy.fa"
svs = "/scratch/ucgd/lustre/u1006375/doubloon/ill-plus-ont.gts.vcf"

window = 5 

from cyvcf2 import VCF
from operator import itemgetter
import toolshed as ts
import sys
import numpy as np
from sklearn.linear_model import LinearRegression

header = "chrom pos depth refbase mismatches pairs softstarts softends hardstarts hardends insertstarts insertends deletions splitters splitters1 mean_insert1 mean_insert2 weird discordant discchrom discchromentropy gc65 gc257 duplicity65 duplicity257 spl".split()

totals = []

vcf = VCF(svs)
gts = []
for variant in vcf:

    left = variant.start
    right = variant.INFO["END"]
    ci = variant.INFO["CIPOS"]
    ce = variant.INFO["CIEND"]
    svlen = variant.INFO["SVLEN"]
    gt = variant.gt_types[0]
    gts.append(gt)
    #mid = svlen / 2

    ls, le = left - window, left + window
    rs, re = right - window, right + window
    #ms, me = mid - window, mid + window

    ends = [left, right]

    combined_vector = []

    for (end, s, e, pos) in [("left", ls, le, left),  ("right", rs, re, right)]:

        region = "%s:%d-%d" % (variant.CHROM, s, e)
        cmd = "|~u6000771/bin/bigly -r {ref} {bam} {region}".format(**locals())

        li = []
        for d in ts.reader(cmd, header=header):
            li.append({
                'softstarts': int(d['softstarts']),
                'softends': int(d['softends']),
                'hardstarts': int(d['hardstarts']),
                'hardends': int(d['hardends']),
                'splitters': int(d['splitters']),
                'depth': int(d['depth'])})

            if len(li) > 1:
                if end == "left":
                    li[-1]['delta'] = li[-2]['depth'] - li[-1]['depth']
                else:
                    li[-1]['delta'] = li[-1]['depth'] - li[-2]['depth']
            else:
                li[0]['delta'] = 0

        li[-1 if end == "left" else 0]['delta'] = 0

        if end == "left":
            suf = 'starts'
        else:
            suf = 'ends'

        for d in li:
            if d['delta'] < 0: d['delta'] = 0
            d['score'] = d['soft' + suf] + d['hard' + suf] + d['delta'] #+ 1.1 * d['deltapairs']
            if d['delta'] == d['score']:
                # depth with no other support gets down-weighted.
                d['score'] = int(d['score'] / 1.5)

#        li.sort(key=itemgetter('score'), reverse=True)
        #if end == "right" and li[0]['score'] > 10:
        #    ends[1] = li[0]['pos']
        #    if abs(ends[1] - variant.INFO["END"]) > 15:
        #        print >> sys.stderr, ends, str(variant)
        #if end == "left" and li[0]['score'] > 10:
        #    ends[0] = li[0]['pos']


        for d in li:
            combined_vector.extend([x[1] for x in d.items()])


    #variant.INFO["END"] = str(ends[1]+1)
    #vstr = str(variant).split("\t")
    #vstr[1] = str(ends[0]+1)
    #print "\t".join(vstr),
    totals.append(combined_vector)

X = np.array(totals)
y = np.array(gts) 
print X
