import argparse
import sys
from pathlib import Path

from runner import optimize_pipeline
from validation import score_and_validate_submission


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    base_dir = Path(__file__).resolve().parent
    default_input = base_dir / 'sample_submission.csv'
    default_output = base_dir / 'submission.csv'

    parser = argparse.ArgumentParser(description="GPU/CPU baseline optimizer and validator")
    parser.add_argument('--mode', choices=['optimize', 'validate'], default='optimize',
                        help="Run optimization or validate an existing submission")
    parser.add_argument('--input', default=str(default_input),
                        help="Input submission CSV (default uses baseline/sample_submission.csv)")
    parser.add_argument('--output', default=str(default_output), help="Output CSV for optimized results")
    parser.add_argument('--limit', type=int, nargs='*', default=None,
                        help="Optional list of N values to process (optimize mode)")
    parser.add_argument('--iters', type=int, default=20, help="Simulated annealing iterations (optimize mode)")
    parser.add_argument('--restarts', type=int, default=10, help="Restart count (optimize mode)")
    parser.add_argument('--fast-tiling-only', action='store_true',
                        help="Skip SA/local search; return deterministic tiling seed (lower CPU use)")
    parser.add_argument('--tune-pair-from', default=None,
                        help="CSV submission used to tune mirror pair offsets (e.g., high-score seed)")
    parser.add_argument('--start-desc', action='store_true',
                        help="Process N in descending order (start from 200)")
    parser.add_argument('--save-each', action='store_true',
                        help="Save CSV after each N is processed")
    parser.add_argument('--jobs', type=int, default=1, help="Parallel workers across N (use 1 if GPU contention)")
    parser.add_argument('--pair-block-x', type=int, default=None, help="GPU pairwise kernel block dim X")
    parser.add_argument('--pair-block-y', type=int, default=None, help="GPU pairwise kernel block dim Y")
    parser.add_argument('--bbox-threads', type=int, default=None, help="GPU bbox kernel threads per block")
    parser.add_argument('--gpu-min-n', type=int, default=None, help="Minimum N to use GPU path")
    parser.add_argument('--force-gpu', action='store_true', help="Force GPU kernels when CUDA is available (ignores gpu-min-n)")
    parser.add_argument('--max-n', type=int, default=200, help="Maximum N to validate (validate mode)")
    args = parser.parse_args()
    if args.mode == 'optimize':
        optimize_pipeline(limit_n=args.limit, input_file=args.input, output_file=args.output,
                  iters=args.iters, restarts=args.restarts, fast_only=args.fast_tiling_only,
                  tune_pair_from=args.tune_pair_from, start_desc=args.start_desc, save_each=args.save_each,
                  jobs=args.jobs,
                  pair_block=(args.pair_block_x, args.pair_block_y) if args.pair_block_x and args.pair_block_y else None,
                  bbox_threads=args.bbox_threads, gpu_min_n=args.gpu_min_n,
                  force_gpu=args.force_gpu)
    else:
        result = score_and_validate_submission(args.input, max_n=args.max_n)
        print(result)


if __name__ == '__main__':
    main()
