from collections import defaultdict
import numpy as np

def calc_depth(bam, chrom, s, e, window=50):
    """
    Calculate the average depth across a region.
    """
    depths = []

    for pileupcolumn in bam.pileup(chrom, s, e):
        start_window = range(s - window, s)
        if pileupcolumn.pos not in start_window:
            continue
        depths.append(pileupcolumn.n)
    return np.mean(depths)
    

def delta_depth(bam, chrom, s, e, window=50):
    """
    Compare the average depth at the start, middle,
    and end of a structural variant. The "delta"
    depth value represents the ratio of depth in the
    middle of a structural variant to the mean depth of the
    ends.
    """

    mid_coord = s + (e - s / 2)
    
    s_depth = calc_depth(bam, chrom, s - window, s)
    mid_depth = calc_depth(bam, chrom, mid_coord - (window / 2), mid_coord + (window / 2))
    e_depth = calc_depth(bam, chrom, e, e + window)

    flank_depth = (s_depth + e_depth) / 2.0

    delta = mid_depth / float(flank_depth) 

    return delta


def alignment_is_clipped(ct, end, tech='ont'):
    """
    If we're looking at alignment at the end of a structural
    variant, check if the first element of a CIGAR string is hard
    or soft clipped. If we're looking at alignments at the
    start of a structural variant, check to see if the last 
    cigar entry is a soft or hard clipped.
    """

    MATCH = 0
    SOFTCLIP = 4
    HARDCLIP = 5

    if tech in ['ont', 'pb']:
        min_clip = 10
    else: min_clip = 3 

    # Look at end of reads if we're at the start of an SV.
    # Look at starts if we're at the end.
    if end == 'left':
        idx = -1
    elif end == 'right':
        idx = 0

    for entry in ct:
        if ct.index(entry) == idx and entry[idx] in [SOFTCLIP, HARDCLIP]:
            clip = entry[1]
            if clip > min_clip:
                return True
        elif ct.index(entry) == idx and entry[idx] == MATCH:
            return False

def count_support(bam, chrom, s, e, end, tech='ont'):
    """
    Given a region, count the number of reference and alternate
    reads. Reads that are split, or clipped over `min_clip` are
    considered ALT support. Reads with at least 90% of bases aligned
    over the region are considered reference.
    """

    ref, alt = 0, 0

    for read in bam.fetch(chrom, s, e):
        overlap = read.get_overlap(s, e)
        if not type(overlap) is int:
            continue
        if overlap / float(e - s) > 0.9:
            ref += 1
            continue
        # If read isn't explicitly labeled as a
        # splitter, check out CIGAR for clips.
        elif alignment_is_clipped(read.cigartuples, end, tech=tech) or read.has_tag('SA'):
            alt += 1
            continue
        else:
            continue

    return (ref, alt)

def flank_is_ref(bam, chrom, s, e, window=25):
    """
    If we're dealing with ONT reads, check to see
    if there is a preponderance of mismatches around
    either end of the SV breakpoints; this would suggest
    that reads are aligning poorly and should be clipped or
    considered alternate support.
    """
    freqs = defaultdict(lambda: defaultdict(int))
    for pileupcolumn in bam.pileup(chrom, s - window, e + window):
        if pileupcolumn.pos < s - window:
            continue
        elif pileupcolumn.pos > e + window:
            break
        for pileupread in pileupcolumn.pileups:
            if pileupread.query_position is None:
                continue
            read_base = pileupread.alignment.query_sequence[pileupread.query_position]
            freqs[pileupcolumn.pos][read_base] += 1

    freq = []
    for i in range(s - window, e + window):
        v = freqs[i]
        if len(v) == 0:
            freq.append(0)
            continue

        top = max(v.values())
        freq.append(top / float(sum(v.values())))

    ref_freq, ref_proportion = 0.8, 0.8

    return sum(f > ref_freq for f in freq) > ref_proportion * len(freq)
