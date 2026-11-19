"""Retired method-specific entry point; preserved to give an explicit migration error."""

def main():
    raise SystemExit('Post-selection policy moved to the algorithm. Use python -m '
                     'rsisys.algorithms.coding_meta_agent.frozen with a new experiment_dir. '
                     'Archived experiment bindings are preserved; do not reuse old run directories.')

if __name__ == '__main__':
    main()
