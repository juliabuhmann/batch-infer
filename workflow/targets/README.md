
- `alphafold3_onegpu` starts from input .json-s in `alphafold3_jsons`, runs data pipeline (one per input), and predictions (all as one GPU job)
- `alphafold3_msas_only` starts from input .json-s in `alphafold3_jsons`, runs data pipeline (one per input)
- `alphafold3_datafill_predictions_batched` like `alphafold3_datafill_predictions` (data-fill per pool on the node, one zip per pool) but N pools per GPU job, packed by a per-engine runtime table (`alphafold3.predictions_batched`); for Anthropic's kits, which take `--input_dir` and amortise start-up/compile over the batch
- `alphafold3_predictions_multigpu` starts from data pipeline output in `alphafold3_msas`, predicts prediction step input times, groups inputs by size into 4-hour group, submits each group as a separate GPU job

- `alphafold3_tests` runs predictions on all available GPU models
