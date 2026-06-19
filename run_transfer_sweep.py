from pathlib import Path
from datetime import datetime
import os, sys, subprocess, shlex, select
import pandas as pd


# Paste/run this cell after setup. It runs the four relation-pressure tests only.
EPOCHS_PER_TEST = 6


TRANSFER_SWEEP = [
    (
        "relation_object_retention6",
        "Sensitivity test: higher object and heatmap token-selection pressure.",
        {
            "--actor_object_relation_null_logit_init": "2.0",
            "--actor_object_relation_geometry_bias_weight": "1.50",
            "--actor_object_relation_heatmap_bias_weight": "3.00",
            "--actor_object_relation_max_scale": "3.00",
            "--token_selection_cls_weight": "0.15",
            "--token_selection_actor_weight": "0.20",
            "--token_selection_object_weight": "0.00",
            "--token_selection_heatmap_weight": "0.30",
            "--actor_object_prompt_box_prior_weight": "0.25",
            "--actor_object_prompt_box_prior_expand": "1.75",
            "--actor_object_relation_loss_weight": "1.25",
        },
    ),
    (
        "relation_conservative6",
        "Range test: weaker relation updates with the same one actor-object relation objective.",
        {
            "--actor_object_relation_null_logit_init": "4.0",
            "--actor_object_relation_geometry_bias_weight": "0.50",
            "--actor_object_relation_heatmap_bias_weight": "1.00",
            "--actor_object_relation_max_scale": "1.00",
            "--token_selection_cls_weight": "0.25",
            "--token_selection_actor_weight": "0.25",
            "--token_selection_object_weight": "0.00",
            "--token_selection_heatmap_weight": "0.30",
            "--actor_object_prompt_box_prior_weight": "0.12",
            "--actor_object_prompt_box_prior_expand": "1.50",
            "--actor_object_relation_loss_weight": "0.80",
        },
    ),
    (
        "relation_strong6",
        "Sensitivity test: stronger in-transformer relation updates without side-head losses.",
        {
            "--actor_object_relation_null_logit_init": "1.75",
            "--actor_object_relation_geometry_bias_weight": "2.00",
            "--actor_object_relation_heatmap_bias_weight": "4.00",
            "--actor_object_relation_max_scale": "4.00",
            "--token_selection_cls_weight": "0.15",
            "--token_selection_actor_weight": "0.20",
            "--token_selection_object_weight": "0.00",
            "--token_selection_heatmap_weight": "0.30",
            "--actor_object_prompt_box_prior_weight": "0.25",
            "--actor_object_prompt_box_prior_expand": "1.75",
            "--actor_object_relation_loss_weight": "1.50",
        },
    ),
    (
        "relation_high_pressure6",
        "High-pressure relation setting for sensitivity checks; do not use as the default run.",
        {
            "--actor_object_relation_null_logit_init": "1.25",
            "--actor_object_relation_geometry_bias_weight": "2.50",
            "--actor_object_relation_heatmap_bias_weight": "5.00",
            "--actor_object_relation_max_scale": "5.00",
            "--token_selection_cls_weight": "0.08",
            "--token_selection_actor_weight": "0.15",
            "--token_selection_object_weight": "0.00",
            "--token_selection_heatmap_weight": "0.25",
            "--actor_object_prompt_box_prior_weight": "0.45",
            "--actor_object_prompt_box_prior_expand": "2.25",
            "--actor_object_relation_loss_weight": "2.00",
        },
    ),
]


if "REPO_DIR" not in globals():
    env_file = Path("/content/poguise_colab_env.sh")
    if not env_file.is_file():
        raise RuntimeError("Run Cell 1 first; /content/poguise_colab_env.sh is missing.")
    for line in env_file.read_text().splitlines():
        if not line.startswith("export ") or "=" not in line:
            continue
        key, value = line[len("export "):].split("=", 1)
        globals()[key] = shlex.split(value)[0]

os.chdir(REPO_DIR)


def run_checked(cmd, label=None):
    print("\n" + "=" * 100, flush=True)
    if label:
        print(label, flush=True)
    print("$ " + " ".join(shlex.quote(str(x)) for x in cmd), flush=True)
    print("=" * 100, flush=True)
    subprocess.run(cmd, cwd=REPO_DIR, check=True)


run_checked([
    sys.executable, "-m", "py_compile",
    "blocks/poguise.py",
    "datasets/object_vocab.py",
    "datasets/toyota_action_taxonomy.py",
    "datasets/toyotasm.py",
    "models/poguise.py",
    "modules/heatmap_module.py",
    "losses/poguiseplus_losses.py",
    "train.py",
    "summarize_interaction_metrics.py",
    "utils/bisect_actor_tensorrt.py",
    "utils/export_actor_tensorrt.py",
    "utils/actor_model.py",
], "Compile check")


def latest_metrics_path(run_dir):
    paths = sorted(Path(run_dir).glob("version_*/metrics.csv"), key=lambda p: p.stat().st_mtime)
    return paths[-1] if paths else None


def completed_validation_epoch_count(metrics_path):
    if metrics_path is None or not Path(metrics_path).is_file():
        return 0
    try:
        raw = pd.read_csv(metrics_path)
    except Exception:
        return 0
    if "epoch" not in raw.columns:
        return 0
    val_cols = [c for c in ["val_loss", "val_f1", "val_acc_macro", "val_deploy_score"] if c in raw.columns]
    if not val_cols:
        return 0

    def last_nonnull(s):
        s = s.dropna()
        return s.iloc[-1] if len(s) else float("nan")

    df = raw.groupby("epoch", as_index=False).agg({c: last_nonnull for c in raw.columns if c != "epoch"})
    df = df.dropna(how="all", subset=val_cols)
    return len(df)


def summarize_run(run_dir, verbose=False):
    cmd = [sys.executable, "summarize_interaction_metrics.py", "--run", str(run_dir)]
    if verbose:
        cmd.append("--verbose")
    print("\n" + "=" * 100, flush=True)
    print("$ " + " ".join(shlex.quote(str(x)) for x in cmd), flush=True)
    print("=" * 100, flush=True)
    result = subprocess.run(cmd, cwd=REPO_DIR, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, flush=True)
    if result.returncode != 0:
        print("Summary not ready yet.", flush=True)


def drain_training_output(proc):
    while True:
        ready, _, _ = select.select([proc.stdout], [], [], 0)
        if not ready:
            break
        line = proc.stdout.readline()
        if not line:
            break
        print(line, end="", flush=True)


def run_training_with_epoch_summaries(cmd, run_name, epoch_dir, poll_secs=20):
    run_dir = Path(DATA_DIR) / "checkpoints" / run_name
    epoch_dir = Path(epoch_dir)
    last_completed_val_count = 0
    last_ckpt_count = 0

    print("\nRUN_NAME:", run_name, flush=True)
    print("RUN_DIR:", run_dir, flush=True)
    print("EPOCH_DIR:", epoch_dir, flush=True)
    print("\n" + "=" * 100, flush=True)
    print("$ " + " ".join(shlex.quote(str(x)) for x in cmd), flush=True)
    print("=" * 100, flush=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        cmd, cwd=REPO_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )

    while True:
        select.select([proc.stdout], [], [], poll_secs)
        drain_training_output(proc)

        ckpt_count = len(sorted(epoch_dir.glob("epoch=*.ckpt")))
        if ckpt_count > last_ckpt_count:
            last_ckpt_count = ckpt_count
            print(f"\nCheckpoint files seen: {ckpt_count}", flush=True)

        metrics_path = latest_metrics_path(run_dir)
        completed_count = completed_validation_epoch_count(metrics_path)
        if completed_count > last_completed_val_count:
            last_completed_val_count = completed_count
            print("\n" + "#" * 100, flush=True)
            print(f"SUMMARY AFTER {completed_count} COMPLETED VALIDATION EPOCH(S): {run_name}", flush=True)
            print("#" * 100, flush=True)
            summarize_run(run_dir, verbose=False)

        if proc.poll() is not None:
            drain_training_output(proc)
            for line in proc.stdout:
                print(line, end="", flush=True)
            break

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)

    print("\nTRAINING DONE:", run_dir, flush=True)
    summarize_run(run_dir, verbose=True)
    return str(run_dir)


def set_cmd_arg(cmd, key, value):
    value = str(value)
    if key in cmd:
        idx = cmd.index(key)
        if idx + 1 >= len(cmd):
            raise RuntimeError(f"{key} is missing a value")
        cmd[idx + 1] = value
    else:
        cmd.extend([key, value])


def make_base_cmd(run_name, epoch_dir):
    return [
        sys.executable, "-u", "train.py",

        "--pretrained", "DEFAULT",
        "--net_size", "b",

        "--dataset", "toyotasm",
        "--dataset_artifact", "toyotasm",
        "--data_dir", DATA_DIR,
        "--toyota_action_taxonomy", "product_v1",
        "--toyota_frame_source", "frames",
        "--toyota_skeleton_zip", SKELETON_ZIP,
        "--toyota_frame_count_cache", FRAME_COUNT_CACHE,
        "--toyota_object_cache_dir", f"{DATA_DIR}/toyota_preprocessed_cache/objects",
        "--toyota_landmark_cache_dir", f"{DATA_DIR}/toyota_preprocessed_cache/landmarks",
        "--toyota_actor_box_jitter_prob", "0.8",
        "--toyota_actor_box_center_jitter", "0.08",
        "--toyota_actor_box_scale_min", "0.9",
        "--toyota_actor_box_scale_max", "1.3",
        "--toyota_split_source", "auto",
        "--toyota_val_fraction", "0.15",
        "--toyota_test_fraction", "0.0",

        "--num_classes", "26",
        "--n_landmarks", "13",
        "--hw_out_conv", "8",

        "--actor_prompt", "1",
        "--num_actor_tokens", "2",
        "--actor_pair_train_weight", "0.50",
        "--actor_presence_head", "1",
        "--presence_loss_weight", "0.05",

        "--actor_interaction_heatmaps", "1",

        "--actor_object_prompt_tokens", "1",
        "--actor_object_relation_in_transformer", "1",
        "--actor_object_relation_blocks", "2,5,8",
        "--actor_object_relation_null_logit_init", "2.0",
        "--actor_object_relation_geometry_bias_weight", "1.5",
        "--actor_object_relation_heatmap_bias_weight", "3.0",
        "--actor_object_relation_max_scale", "3.0",
        "--actor_object_relation_learned_scale", "1",
        "--actor_object_relation_layer_scale_init", "0.25",
        "--actor_relation_action_fusion", "1",
        "--token_selection_cls_weight", "0.20",
        "--token_selection_actor_weight", "0.20",
        "--token_selection_object_weight", "0.00",
        "--token_selection_heatmap_weight", "0.30",
        "--actor_object_prompt_box_prior_weight", "0.18",
        "--actor_object_prompt_box_prior_expand", "1.60",
        "--num_scene_object_tokens", "32",
        "--num_object_classes", "19",

        "--object_detector_cache", OBJECT_DETECTOR_CACHE,
        "--object_camera_allowlist", "tv_monitor=c05,c06",
        "--object_ignore_regions", "c03=0,0,0.26,0.42",
        "--object_conf_threshold", "0.25",

        "--interaction_heatmap_sigma", "2.5",

        "--class_balanced_sampler", "1",

        "--keep_rate", "0.6",
        "--keep_rate_merge", "0.3",
        "--merge_type", "sim",
        "--merge_mode", "0",
        "--sim_metric", "1",
        "--topk_type", "1",

        "--grad_weights", "1",
        "--nash_update_weights_every", "20",
        "--nash_max_norm", "2.0",

        "--poguiseplus_heatmap_loss_weight", "1.0",
        "--poguiseplus_pose_heatmap_weight", "0.25",
        "--poguiseplus_interaction_heatmap_weight", "3.0",
        "--poguiseplus_heatmap_log_eps", "1e-6",
        "--poguiseplus_normalized_heatmap_loss", "1",
        "--poguiseplus_heatmap_mse_scale", "1000",

        "--actor_object_relation_loss_weight", "1.00",
        "--actor_object_relation_null_loss_weight", "1.00",


        "--batch_size", "32",
        "--accum_grad_batches", "2",

        "--max_epochs", "10",
        "--t_max_scheduler", "10",

        "--lr", "3e-5",
        "--lr_head", "5e-4",
        "--lr_head_hm", "5e-4",

        "--weight_decay", "0.04",
        "--weight_decay_head", "0.01",
        "--weight_decay_head_hm", "0.005",
        "--label_smoothing", "0.0",
        "--gradient_clip_val", "1.0",

        "--num_workers", "12",
        "--persistent_workers", "1",
        "--prefetch_factor", "2",
        "--precision", "bf16-mixed",
        "--accelerator", "gpu",
        "--gpus", "1",

        "--limit_val_batches", "1.0",
        "--check_val_every_n_epoch", "1",
        "--num_sanity_val_steps", "0",
        "--log_every_n_steps", "50",

        "--checkpoint_monitor", "val_deploy_score",
        "--checkpoint_mode", "max",
        "--checkpoint_filename", "{epoch:03d}-{val_deploy_score:.4f}-{val_f1:.4f}-{val_acc_macro:.4f}",
        "--save_top_k", "5",
        "--save_every_epoch_checkpoints", "1",
        "--epoch_checkpoint_dir", str(epoch_dir),
        "--epoch_checkpoint_filename", "{epoch:03d}",

        "--default_root_dir", f"{DATA_DIR}/checkpoints",
        "--model_name", run_name,
    ]


def run_transfer_test(preset_name, description, overrides):
    run_name = f"actor_object_{preset_name}_{TS}"
    epoch_dir = Path(DATA_DIR) / "checkpoints" / run_name / "epoch_checkpoints"
    cmd = make_base_cmd(run_name, epoch_dir)
    set_cmd_arg(cmd, "--max_epochs", str(EPOCHS_PER_TEST))
    set_cmd_arg(cmd, "--t_max_scheduler", str(EPOCHS_PER_TEST))
    set_cmd_arg(cmd, "--save_top_k", "2")
    set_cmd_arg(cmd, "--save_every_epoch_checkpoints", "0")
    for key, value in overrides.items():
        set_cmd_arg(cmd, key, value)

    print("\n" + "=" * 100, flush=True)
    print(f"STARTING SWEEP RUN: {run_name}", flush=True)
    print(description, flush=True)
    print("OVERRIDES:", flush=True)
    for key, value in overrides.items():
        print(f"  {key} {value}", flush=True)
    print("=" * 100, flush=True)

    return run_training_with_epoch_summaries(
        cmd,
        run_name,
        epoch_dir,
        poll_secs=20,
    )


TS = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dirs = []

print("\nTRANSFER OBJECT-RELATION SWEEP", flush=True)
print(f"epochs per run: {EPOCHS_PER_TEST}", flush=True)
print("This runs four tests only. It does not run a separate 10-epoch baseline.", flush=True)

for preset_name, description, overrides in TRANSFER_SWEEP:
    run_dirs.append(run_transfer_test(preset_name, description, overrides))

print("\nSWEEP DONE", flush=True)
for run_dir in run_dirs:
    print("RUN_DIR:", run_dir, flush=True)
