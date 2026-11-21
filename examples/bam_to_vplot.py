#!/usr/bin/env python
"""BAM + motif BED -> cached store, QC, V-plot, footprint, scores. Same as the `pyvplot-report` command.

    python bam_to_vplot.py --bam sample.bam --motifs motifs.bed --out results/sample

The pipeline lives in `pyvplot/cli.py`; copy that file to adapt it.
"""

from pyvplot.cli import main

if __name__ == "__main__":
    main()
