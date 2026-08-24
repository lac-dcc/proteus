#!/usr/bin/env bash
set -euo pipefail

mkdir -p csv_data
if [ ! -f csv_data/zero-counts.csv ]; then \
    LATTICE_FILTER=banded-16,banded-32,banded-64,banded-128,banded-192,all-sparse \
    ./scripts/run_zero_counts.sh --csv > csv_data/zero-counts.csv; \
fi
if [ ! -f csv_data/mbench-timings.csv ]; then \
    LATTICE_FILTER=banded-16,banded-32,banded-64,banded-128,banded-192,all-sparse \
    MODEL_FILTER=mbench_conv2d,mbench_depthwise_conv2d,mbench_pooling_max,mbench_pooling_sum \
    NO_O3=1 \
    ./scripts/run_timings.sh --csv > csv_data/mbench-timings.csv; \
fi
if [ ! -f csv_data/timings.csv ]; then \
    LATTICE_FILTER=banded-16,banded-32,banded-64,banded-128,banded-192,all-sparse \
    ./scripts/run_timings.sh --csv > csv_data/timings.csv; \
fi
if [ ! -f csv_data/compile_timings.csv ]; then \
    LATTICE_FILTER=banded-64,banded-128,banded-192,all-sparse \
    ./scripts/run_compile_times.sh --csv > csv_data/compile_timings.csv; \
fi
python3 scripts/plot_breakoff.py
python3 scripts/plot_breakeven.py
python3 scripts/plot_zero_counts.py
python3 scripts/plot_mbench_timings.py
python3 scripts/plot_runtimes.py
python3 scripts/plot_compile_times.py
