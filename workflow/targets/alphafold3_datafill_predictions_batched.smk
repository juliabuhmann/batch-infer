"""
alphafold3_datafill_predictions with N pools per GPU job.

Same flow as alphafold3_datafill_predictions.smk (af3io data-fill on the node, one container call, one zip per pool) but
the pools are packed into jobs by predicted runtime instead of one pool per job. The rule body there was already
written for a batch (loops over ALPHAFOLD3_IDS, --input_dir); only its batch function was a placeholder ("Mock
partitioning to run one pool per job for now").

Written for Anthropic's af3_torch optimization kit (run_alphafold_wrapper + singularity_args of the run config; the kit
loads the model once per process and overlaps featurisation/writing of one item with the GPU on another, measured
1.27x more throughput at 3000 tokens with 3 items per call, and pays its ~60 s cold compile once per job instead of
once per pool). Engine-agnostic otherwise: the container takes --input_dir like stock run_alphafold.py.

Config (workflow/config/defaults.yaml `alphafold3.predictions_batched`, overridable per run):
  runtime            SLURM time limit of a batch job
  budget_hrs         sum of predicted pool runtimes per job (plus job_overhead_s) -- keep below `runtime`: a job cut by
                     its time limit loses the pool in flight and every pool not yet zipped (see "zips" below)
  job_overhead_s     once per job: container start, imports, model load, cold compile
  max_pools_per_job  hard cap on the batch length
  tokens_min/max     pools outside are refused (listed in <tsv>.refused) -- e.g. the kit's memory reach on a card
  order              tokens_desc (default: the largest pool first, so an out-of-memory pool fails in the first
                     minutes and like-sized pools share a compile) | tokens_asc | none (id order)
  runtime_table_s    [[tokens, seconds], ...]: per-pool wall time of the engine in a multi-pool process, linearly
                     interpolated (flat outside the table) -- measure it per engine/card
  disk_mb_per_pool, disk_mb_min   local scratch: filled input (~0.3 GB) + outputs (~1.1 GB) + caches, per pool
  early_zip          true (default) | false: zip only after the container exits, as the first batched runs did
  threads            CPU cores of a batch job (sbatch --ntasks; mem_mb is split over them). The one-pool target runs
                     on 1 core; the kit runs featurisation and the write-behind writer as separate processes beside
                     the model process, and the early zipper below adds one more -- 4 by default (gain UNMEASURED).

Batches are cached in alphafold3_predictions/.alphafold3_datafill_predictions_batched.tsv on first use, like the
one-pool target's list: delete it (or re-run the staging stage) after inputs or zips change. The next `batch-infer
start` then re-batches the pools that still have no zip.

Zips: a pool is zipped AS SOON AS its output is complete (a watcher beside the container zips every pool whose
<id>_ranking_scores.csv -- the last file AlphaFold 3's write_outputs writes -- has been untouched for
ZIP_SETTLE_S seconds), and once more for whatever is left after the container exits. So a job that hits its time
limit keeps the pools it finished; only the pool in flight (and the ones the watcher had not reached) are lost and
re-batched by the next run. Pools that already have a zip when the job starts (a stale batch list) are skipped.

Per-pool failure: the container exits non-zero when any item failed (Anthropic's kits account every item and go on
to the next; stock run_alphafold.py stops at the first). The rule tolerates that; a pool without a zip is what marks
a failure, and re-running the target re-batches the pools still missing a zip -- batch-infer's resume model. The
rule's declared output is therefore a per-batch marker (alphafold3_predictions/.batched/batch<N>_<digest>.done,
listing each pool's status), NOT the zips: snakemake removes the outputs of a failed job, which would delete the
finished pools' zips along with the failed pool's. <digest> is a hash of the batch's pool ids, so a batch list
regenerated after a partial run (batch numbers start at 0 again, over other pools) never matches an old marker.

STATUS 2026-09-30: batch packing and the one-container-per-batch flow ran (8 pools, job 15714071; 1 + 19 + 19 pools,
jobs 15721990/88/89). The early zipper, the zip-exists skip, the digest markers and `threads` are NOT yet run.
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


def batch_marker_(batch_id, ids_):
    """alphafold3_predictions/.batched/batch<N>_<digest>.done -- the digest ties the marker to the batch's pool ids."""
    import hashlib
    digest_ = hashlib.sha1(' '.join(ids_).encode()).hexdigest()[:8]
    return f'{MARK_DIR_}/batch{batch_id}_{digest_}.done'


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
markers_ = []

for batch_id, df_batch in batches_.groupby('batch_id'):
    n_pools_ = len(df_batch)
    marker_ = batch_marker_(batch_id, df_batch.id.tolist())
    markers_.append(marker_)

    rule:  # N pools of the batch: fill all, one container call over all, one zip per pool (early, then the rest)
        name:
            f'alphafold3_datafill_predictions_batched{batch_id}_{n_pools_}'
        input:
            json = expand('alphafold3_jsons/{id}.json', id=df_batch.id.tolist()),
        output:
            done = marker_,
        params:
            activate = root_path('.venv/bin/activate'),
            alphafold3_ids = ' '.join(df_batch.id.tolist()),
            est_min = round(df_batch.runtime_s.sum() / 60, 1),
            zip_settle_s = int(CFG_B.get('zip_settle_s', 30)),
            early_zip = int(bool(CFG_B.get('early_zip', True))),
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
        threads: int(CFG_B.get('threads', 4))
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
            MARKER_TMP=$SMKDIR/{output.done}.tmp
            : > $MARKER_TMP

            # pools that already have a zip (a batch list older than the zips) are not run again
            TODO_IDS=""
            for ALPHAFOLD3_ID in {params.alphafold3_ids}; do
                if [ -f $SMKDIR/alphafold3_predictions/"$ALPHAFOLD3_ID".zip ]; then
                    echo "POOL_STATUS $ALPHAFOLD3_ID already_zipped" | tee -a $MARKER_TMP
                else
                    TODO_IDS="$TODO_IDS $ALPHAFOLD3_ID"
                fi
            done
            if [ -z "$TODO_IDS" ]; then
                echo BATCH_DONE: ok=0 of 0 "(every pool already zipped)"
                mv $MARKER_TMP $SMKDIR/{output.done}
                exit 0
            fi

            mkdir -p $TMPDIR/alphafold3_msas $TMPDIR/alphafold3_jsons_batch
            for ALPHAFOLD3_ID in $TODO_IDS; do
                echo ALPHAFOLD3_ID: "$ALPHAFOLD3_ID"
                ln -sfn $SMKDIR/alphafold3_jsons/"$ALPHAFOLD3_ID".json $TMPDIR/alphafold3_jsons_batch/"$ALPHAFOLD3_ID".json
            done
            # ONE af3io call for the whole batch (--input_dir over the staged links): the data index is built once per
            # job, not once per pool -- a cache directory without .af3io_data_index.json is re-read on every call
            # (688 files, ~1 min each in the first batched run), which the one-pool target hides and a batch multiplies.
            echo FILL_START: `date -Is`
            af3io data-fill {params.data_sources} --data_dir=alphafold3_msas --input_dir=$TMPDIR/alphafold3_jsons_batch --output_dir=$TMPDIR/alphafold3_msas/
            echo FILL_END: `date -Is`
            ls -l $TMPDIR/alphafold3_msas

            mkdir -p $TMPDIR/alphafold3_predictions
            mkdir -p $TMPDIR/alphafold_cache
            cd $TMPDIR
            echo Contents of $TMPDIR
            ls -l $TMPDIR

            # zip_pool <id>: cwd $TMPDIR/alphafold3_predictions; the filled input copy is dropped, the zip lands atomically
            zip_pool() {{
                [ -f "$1"/"$1"_data.json ] && rm "$1"/"$1"_data.json
                zip -q -r $SMKDIR/alphafold3_predictions/"$1".zip.tmp "$1" \\
                    && mv $SMKDIR/alphafold3_predictions/"$1".zip.tmp $SMKDIR/alphafold3_predictions/"$1".zip
            }}
            # early zipper: beside the container, a pool is zipped once its <id>_ranking_scores.csv (the last file of
            # AlphaFold 3's write_outputs) has sat untouched for zip_settle_s; ZIPPED_EARLY lines name them
            ZIPPED_LIST=$TMPDIR/zipped_early.txt
            : > $ZIPPED_LIST
            ( set +e; cd $TMPDIR/alphafold3_predictions
              [ "{params.early_zip}" = 1 ] || exit 0    # early_zip: false -> zip everything after the container (control runs)
              while true; do
                  for ALPHAFOLD3_ID in $TODO_IDS; do
                      grep -qx "$ALPHAFOLD3_ID" $ZIPPED_LIST && continue
                      RANK="$ALPHAFOLD3_ID"/"$ALPHAFOLD3_ID"_ranking_scores.csv
                      [ -f "$RANK" ] && [ -f "$ALPHAFOLD3_ID"/"$ALPHAFOLD3_ID"_summary_confidences.json ] || continue
                      [ $(( `date +%s` - `stat -c %Y "$RANK"` )) -ge {params.zip_settle_s} ] || continue
                      if zip_pool "$ALPHAFOLD3_ID"; then
                          echo "$ALPHAFOLD3_ID" >> $ZIPPED_LIST
                          echo ZIPPED_EARLY: "$ALPHAFOLD3_ID" `date -Is`
                      fi
                  done
                  [ -f $TMPDIR/container.finished ] && break
                  sleep 30
              done ) &
            ZIPPER_PID=$!

            echo CONTAINER_START: `date -Is`
            CONTAINER_RC=0
            singularity exec --nv --writable-tmpfs \\
                {params.singularity_args} \\
                {params.bind_input} \\
                {params.bind_output} \\
                {params.bind_models} \\
                {params.bind_databases} \\
                {params.bind_scripts} \\
                {params.container} \\
                sh -c '/app/scripts/{params.run_alphafold_wrapper} {params.run_alphafold_args} {params.run_alphafold_dirs}' || CONTAINER_RC=$?
            echo CONTAINER_END: `date -Is` rc=$CONTAINER_RC
            touch $TMPDIR/container.finished
            wait $ZIPPER_PID || true
            cd -

            # the rest: one zip per pool that produced a summary and was not zipped early; a pool without a summary
            # stays without a zip (= failed, re-batched by the next run). The batch marker lists every pool's status.
            cd $TMPDIR/alphafold3_predictions
            N_OK=0
            for ALPHAFOLD3_ID in $TODO_IDS; do
                if grep -qx "$ALPHAFOLD3_ID" $ZIPPED_LIST; then
                    N_OK=$((N_OK + 1)); echo "POOL_STATUS $ALPHAFOLD3_ID ok" >> $MARKER_TMP
                elif [ -f "$ALPHAFOLD3_ID"/"$ALPHAFOLD3_ID"_summary_confidences.json ]; then
                    echo Compressing predictions to $SMKDIR/alphafold3_predictions/$ALPHAFOLD3_ID.zip
                    if zip_pool "$ALPHAFOLD3_ID"; then
                        N_OK=$((N_OK + 1)); echo "POOL_STATUS $ALPHAFOLD3_ID ok" >> $MARKER_TMP
                    else
                        echo "POOL_STATUS $ALPHAFOLD3_ID zip_failed" | tee -a $MARKER_TMP
                    fi
                else
                    echo "POOL_STATUS $ALPHAFOLD3_ID missing_output" | tee -a $MARKER_TMP
                fi
            done
            echo BATCH_DONE: ok=$N_OK of `echo $TODO_IDS | wc -w` container_rc=$CONTAINER_RC early=`wc -l < $ZIPPED_LIST`
            if [ "$N_OK" -eq 0 ] && [ "$CONTAINER_RC" -ne 0 ]; then
                echo "ERROR: the container failed (rc $CONTAINER_RC) and no pool produced output" >&2
                rm -f $MARKER_TMP
                exit $CONTAINER_RC
            fi
            mv $MARKER_TMP $SMKDIR/{output.done}
        """

rule alphafold3_datafill_predictions_batched:
    input:
        markers_,
