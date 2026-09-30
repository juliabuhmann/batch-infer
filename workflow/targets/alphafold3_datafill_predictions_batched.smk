"""
alphafold3_datafill_predictions with N pools per GPU job.

Same flow as alphafold3_datafill_predictions.smk (af3io data-fill per pool on the node, one container call, one zip per
pool) but the pools are packed into jobs by predicted runtime instead of one pool per job. The rule body there was
already written for a batch (loops over ALPHAFOLD3_IDS, --input_dir); only its batch function was a placeholder
("Mock partitioning to run one pool per job for now").

Written for Anthropic's af3_torch optimization kit (run_alphafold_wrapper + singularity_args of the run config; the kit
loads the model once per process and overlaps featurisation/writing of one item with the GPU on another, measured
1.27x more throughput at 3000 tokens with 3 items per call, and pays its ~60 s cold compile once per job instead of
once per pool). Engine-agnostic otherwise: the container takes --input_dir like stock run_alphafold.py.

Config (workflow/config/defaults.yaml `alphafold3.predictions_batched`, overridable per run):
  runtime            SLURM time limit of a batch job
  budget_hrs         sum of predicted pool runtimes per job (plus job_overhead_s) -- keep well below `runtime`:
                     the zips are written only after the container exits, so a job that hits its time limit loses
                     EVERY pool of the batch (its outputs sit on the node's local scratch)
  job_overhead_s     once per job: container start, imports, model load, cold compile
  max_pools_per_job  hard cap on the batch length
  tokens_min/max     pools outside are refused (listed in <tsv>.refused) -- e.g. the kit's memory reach on a card
  order              tokens_desc (default: the largest pool first, so an out-of-memory pool fails in the first
                     minutes and like-sized pools share a compile) | tokens_asc | none (id order)
  runtime_table_s    [[tokens, seconds], ...]: per-pool wall time of the engine in a multi-pool process, linearly
                     interpolated (flat outside the table) -- measure it per engine/card
  disk_mb_per_pool, disk_mb_min   local scratch: filled input (~0.3 GB) + outputs (~1.1 GB) + caches, per pool

Batches are cached in alphafold3_predictions/.alphafold3_datafill_predictions_batched.tsv on first use, like the
one-pool target's list: delete it (or re-run the staging stage) after inputs or zips change.

Per-pool failure: the container exits non-zero when any item failed (Anthropic's kits do; stock run_alphafold.py
stops at the first). The rule tolerates that and zips every pool that has a summary; a pool without a zip is what
marks a failure, and re-running the target re-batches the pools still missing a zip -- batch-infer's resume model.
The rule's declared output is therefore a per-batch marker (alphafold3_predictions/.batched/batch<N>.done, listing
each pool's status), NOT the zips: snakemake removes the outputs of a failed job, which would delete the finished
pools' zips along with the failed pool's.

STATUS 2026-09-30: written on the branch add-anthropic-kit-support; not yet run.
"""

include: '../rules/common.smk'

CFG_B = config['alphafold3']['predictions_batched']
BATCH_TSV_ = 'alphafold3_predictions/.alphafold3_datafill_predictions_batched.tsv'
MARK_DIR_ = 'alphafold3_predictions/.batched'


def pool_tokens_(path):
    """Protein tokens of an AF3 input JSON (sequence-only or filled): residues x copies; the pooled schemes tile
    proteins only. `id` is a string for one chain or a list for copies."""
    with open(path) as fh:
        js = json.load(fh)
    n = 0
    for seq in js['sequences']:
        if 'protein' in seq:
            ids = seq['protein']['id']
            copies = len(ids) if isinstance(ids, list) else 1
            n += copies * len(seq['protein']['sequence'])
    return n


def runtime_s_(tokens):
    tab = np.array(CFG_B['runtime_table_s'], dtype=float)
    tab = tab[np.argsort(tab[:, 0])]
    return float(np.interp(tokens, tab[:, 0], tab[:, 1]))


def make_runtime_batches_():
    os.makedirs(MARK_DIR_, exist_ok=True)
    if not os.path.isfile(BATCH_TSV_):
        import snakemake.io
        ids_jsons_, = snakemake.io.glob_wildcards('alphafold3_jsons/{id}.json')
        ids_infer_, = snakemake.io.glob_wildcards('alphafold3_predictions/{id}.zip')
        todo_ = sorted(set(ids_jsons_) - set(ids_infer_))
        frame_ = pd.DataFrame({'id': todo_})
        frame_['tokens'] = frame_['id'].map(lambda i: pool_tokens_(f'alphafold3_jsons/{i}.json'))
        ok_ = (frame_['tokens'] >= CFG_B['tokens_min']) & (frame_['tokens'] <= CFG_B['tokens_max'])
        if (~ok_).any():
            frame_[~ok_].to_csv(BATCH_TSV_ + '.refused', sep='\t', index=False)
            print(f"{(~ok_).sum()} pool(s) outside tokens [{CFG_B['tokens_min']}, {CFG_B['tokens_max']}] refused -> {BATCH_TSV_}.refused")
        frame_ = frame_[ok_].copy()
        order_ = CFG_B.get('order', 'tokens_desc')
        if order_ == 'tokens_desc':
            frame_ = frame_.sort_values(['tokens', 'id'], ascending=[False, True])
        elif order_ == 'tokens_asc':
            frame_ = frame_.sort_values(['tokens', 'id'])
        frame_ = frame_.reset_index(drop=True)
        frame_['runtime_s'] = frame_['tokens'].map(runtime_s_)
        budget_s_ = float(CFG_B['budget_hrs']) * 3600 - float(CFG_B['job_overhead_s'])
        max_n_ = int(CFG_B['max_pools_per_job'])
        batch_, acc_, n_, bid_ = [], 0.0, 0, 0
        for rt_ in frame_['runtime_s']:
            if n_ and (acc_ + rt_ > budget_s_ or n_ >= max_n_):
                bid_ += 1; acc_ = 0.0; n_ = 0
            batch_.append(bid_); acc_ += rt_; n_ += 1
        frame_['batch_id'] = batch_
        frame_[['id', 'batch_id', 'tokens', 'runtime_s']].to_csv(BATCH_TSV_, sep='\t', header=True, index=False)
        if len(frame_):
            summary_ = frame_.groupby('batch_id').agg(pools=('id', 'size'), tokens_max=('tokens', 'max'),
                                                      est_min=('runtime_s', lambda s: round(s.sum() / 60, 1)))
            print(f"{len(frame_)} pool(s) in {len(summary_)} batch(es), budget {CFG_B['budget_hrs']} h "
                  f"(+{CFG_B['job_overhead_s']} s overhead), order {order_}:")
            print(summary_.to_string())
    return pd.read_csv(BATCH_TSV_, sep='\t', dtype={'id': str})


batches_ = make_runtime_batches_()
ids = batches_.id.tolist()

for batch_id, df_batch in batches_.groupby('batch_id'):
    n_pools_ = len(df_batch)

    rule:  # N pools of the batch: fill each, one container call over all, one zip per pool
        name:
            f'alphafold3_datafill_predictions_batched{batch_id}_{n_pools_}'
        input:
            json = expand('alphafold3_jsons/{id}.json', id=df_batch.id.tolist()),
        output:
            done = f'{MARK_DIR_}/batch{batch_id}.done',
        params:
            activate = root_path('.venv/bin/activate'),
            alphafold3_ids = ' '.join(df_batch.id.tolist()),
            est_min = round(df_batch.runtime_s.sum() / 60, 1),
            data_sources = config['alphafold3']['data_sources'],
            singularity_args = config['alphafold3']['predictions']['singularity_args'],
            container = config['alphafold3']['container'],
            bind_input = '--bind alphafold3_msas:/root/af_input',
            bind_output = '--bind alphafold3_predictions:/root/af_output',
            bind_models = f"--bind {config['alphafold3']['model_dir']}:/root/models",
            bind_databases = f"--bind {config['alphafold3']['db_dir']}:/root/public_databases",
            bind_scripts = f"--bind {root_path('workflow/scripts')}:/app/scripts",
            run_alphafold_wrapper = config['alphafold3']['predictions']['run_alphafold_wrapper'],
            run_alphafold_args = f"--norun_data_pipeline {config['alphafold3']['predictions']['run_alphafold_args']}",
            run_alphafold_dirs = '--input_dir=/root/af_input --output_dir=/root/af_output --model_dir=/root/models --db_dir=/root/public_databases',
        retries: config['alphafold3']['predictions']['retries']
        resources:
            runtime = CFG_B['runtime'],
            mem_mb = config['alphafold3']['predictions']['mem_mb'],
            disk_mb = max(int(CFG_B['disk_mb_min']), int(CFG_B['disk_mb_per_pool']) * n_pools_),
            slurm_extra = config['alphafold3']['predictions']['slurm_extra'],
        envmodules: *config['envmodules_offline']
        shell: """
            source {params.activate}
            SMKDIR=`pwd`
            echo BATCH: {params.alphafold3_ids}
            echo BATCH_EST_MIN: {params.est_min}
            mkdir -p $TMPDIR/alphafold3_msas
            ALPHAFOLD3_IDS="{params.alphafold3_ids}"
            for ALPHAFOLD3_ID in $ALPHAFOLD3_IDS; do
                echo ALPHAFOLD3_ID: "$ALPHAFOLD3_ID"
                JSON=alphafold3_jsons/"$ALPHAFOLD3_ID".json
                echo JSON: "$JSON"
                OUTPUT_DIR="$TMPDIR"/alphafold3_msas/
                echo OUTPUT_DIR: "$OUTPUT_DIR"
                af3io data-fill {params.data_sources} --data_dir=alphafold3_msas --json_path="$JSON" --output_dir="$OUTPUT_DIR"
            done
            ls -l $TMPDIR/alphafold3_msas

            mkdir -p $TMPDIR/alphafold3_predictions
            mkdir -p $TMPDIR/alphafold_cache
            cd $TMPDIR
            echo Contents of $TMPDIR
            ls -l $TMPDIR
            echo CONTAINER_START: `date -Is`
            CONTAINER_RC=0
            singularity exec --nv --writable-tmpfs \
                {params.singularity_args} \
                {params.bind_input} \
                {params.bind_output} \
                {params.bind_models} \
                {params.bind_databases} \
                {params.bind_scripts} \
                {params.container} \
                sh -c '/app/scripts/{params.run_alphafold_wrapper} {params.run_alphafold_args} {params.run_alphafold_dirs}' || CONTAINER_RC=$?
            echo CONTAINER_END: `date -Is` rc=$CONTAINER_RC
            cd -

            # one zip per pool that produced a summary; a pool without one stays without a zip (= failed, re-batched
            # by the next run). The batch marker lists every pool's status.
            cd $TMPDIR/alphafold3_predictions
            N_OK=0
            : > $SMKDIR/{output.done}.tmp
            for ALPHAFOLD3_ID in $ALPHAFOLD3_IDS; do
                if [ -f "$ALPHAFOLD3_ID"/"$ALPHAFOLD3_ID"_summary_confidences.json ]; then
                    OUT_DATA="$ALPHAFOLD3_ID"/"$ALPHAFOLD3_ID"_data.json
                    [ -f "$OUT_DATA" ] && rm "$OUT_DATA"
                    echo Compressing predictions to $SMKDIR/alphafold3_predictions/$ALPHAFOLD3_ID.zip
                    zip -q -r $SMKDIR/alphafold3_predictions/$ALPHAFOLD3_ID.zip.tmp $ALPHAFOLD3_ID \
                        && mv $SMKDIR/alphafold3_predictions/$ALPHAFOLD3_ID.zip.tmp $SMKDIR/alphafold3_predictions/$ALPHAFOLD3_ID.zip \
                        && N_OK=$((N_OK + 1)) && echo "POOL_STATUS $ALPHAFOLD3_ID ok" >> $SMKDIR/{output.done}.tmp \
                        || echo "POOL_STATUS $ALPHAFOLD3_ID zip_failed" >> $SMKDIR/{output.done}.tmp
                else
                    echo "POOL_STATUS $ALPHAFOLD3_ID missing_output" | tee -a $SMKDIR/{output.done}.tmp
                fi
            done
            echo BATCH_DONE: ok=$N_OK of `echo $ALPHAFOLD3_IDS | wc -w` container_rc=$CONTAINER_RC
            if [ "$N_OK" -eq 0 ] && [ "$CONTAINER_RC" -ne 0 ]; then
                echo "ERROR: the container failed (rc $CONTAINER_RC) and no pool produced output" >&2
                rm -f $SMKDIR/{output.done}.tmp
                exit $CONTAINER_RC
            fi
            mv $SMKDIR/{output.done}.tmp $SMKDIR/{output.done}
        """

rule alphafold3_datafill_predictions_batched:
    input:
        expand(f'{MARK_DIR_}/batch{{batch_id}}.done', batch_id=sorted(batches_.batch_id.unique())),
