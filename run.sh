python tr.py --bam ../sv_sensitivity_ont/nanopore-NA12878/nanopore-NA12878.mmp.merged.bam --vcf ill-plus-ont.gts.vcf -tech ont
python tr.py --bam ../sv_sensitivity_ont/nanopore-NA12878/nanopore-NA12878.mmp.merged.bam --vcf ill-plus-ont.gts.vcf -tech ont -training

python svelte.py
