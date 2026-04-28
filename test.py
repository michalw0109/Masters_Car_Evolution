"""
Test script for evaluating trained car evolution models.

Usage:
    python test.py <parent_dir>

<parent_dir> must contain exactly 4 sub-folders, one per seed.
Each experiment directory must contain:
    - bestValDNA.txt  : DNA string of the best model
    - graphData.txt   : columns: epoch best_train avg_train val_fitness

assets/ must contain:
    - test1.png, test2.png, ...  (at least one)
    - parameters.txt with CAR_TEST1_X/Y/A, CAR_TEST2_X/Y/A, ... entries
"""

import sys
import os
import csv
import pygame
import numpy as np
from Colors import Color
from Car import Car
from Computation import Computation
from DNA_Decoder import Decoder
from DNA import Single_DNA_one_chromosome


# ── config ────────────────────────────────────────────────────────────────────
INPUTS = [0, 1, 2, 3, 4, 5]
OUTPUTS = [12, 13, 14, 15]
MARKER = [0, 1, 1, 1, 1, 1, 1, 1]

parent_dir = R"C:\Users\micha\Desktop\studia\magisterka\Car_evolution\experiements\single_dna\init\single_dna_cells_init"
DNA_DECODER = Decoder().decodes_single_DNA_one_chromosome(INPUTS, OUTPUTS, MARKER).cellular_division

COMPUTATION = Computation(INPUTS, OUTPUTS, MARKER).connection_based_sort_feed_forward

FPS            = 1200
PACE_THRESHOLD = 100.0  # fitness for 1 full lap
LAP_THRESHOLD  = 100.0  # minimum score to count a sim as successful


# ── data structures ───────────────────────────────────────────────────────────

class TrackResult:
    """Result of one car run on one test track."""
    def __init__(self, track_idx: int, score: float, speed_log: list):
        self.track_idx = track_idx
        self.score     = score
        self.speed_log = speed_log
        self.made_lap  = score >= LAP_THRESHOLD


class SimData:
    """All data for one simulation seed."""
    def __init__(self, sim_idx: int, exp_dir: str,
                 graph_data: list, track_results: list):
        self.sim_idx       = sim_idx
        self.exp_dir       = exp_dir
        self.graph_data    = graph_data     # [(epoch, best_train, avg_train, val)]
        self.track_results = track_results  # [TrackResult]


class ExperimentData:
    """Full collected data for an experiment (4 seeds × N tracks)."""
    def __init__(self, sims: list, track_names: list):
        self.sims        = sims         # [SimData]
        self.track_names = track_names  # ['test1.png', ...]


# ── file I/O ──────────────────────────────────────────────────────────────────

def load_params(path: str) -> dict:
    params = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, value = line.split(maxsplit=1)
            if value.lstrip('-').isdigit():
                params[key] = int(value)
            elif value.lstrip('-').replace('.', '', 1).isdigit():
                params[key] = float(value)
            elif value.lower() in ('true', 'false'):
                params[key] = value.lower() == 'true'
            else:
                params[key] = value
    return params


def load_dna(path: str) -> list:
    with open(path, 'r') as f:
        return [int(c) for c in f.readline().strip()]


def load_graph_data(path: str) -> list:
    rows = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                rows.append((int(parts[0]), float(parts[1]),
                              float(parts[2]), float(parts[3])))
    return rows


# ── track loading ─────────────────────────────────────────────────────────────

def load_test_tracks(params: dict, W: int, H: int) -> list:
    """
    Detect assets/test1.png, test2.png, ... and load each one.
    Each track requires CAR_TEST{n}_X, CAR_TEST{n}_Y, CAR_TEST{n}_A in params.
    """
    tracks = []
    i = 1
    while True:
        path  = f"assets/test_track{i}.png"
        x_key = f"CAR_TEST_X{i}"
        if not os.path.exists(path) or x_key not in params:
            break
        surface = pygame.image.load(path)
        surface = pygame.transform.scale(surface, (W, H))
        pixels  = pygame.surfarray.pixels3d(surface)
        cmap    = np.all(pixels == Color.GREEN, axis=2)
        del pixels
        tracks.append({
            'idx':           i,
            'name':          f"test{i}.png",
            'surface':       surface,
            'collision_map': cmap,
            'start':         [params[f"CAR_TEST_X{i}"], params[f"CAR_TEST_Y{i}"]],
            'angle':         params[f"CAR_TEST_A{i}"],
        })
        i += 1
    return tracks


# ── simulation ────────────────────────────────────────────────────────────────

def run_simulation(dna_bits: list, screen, track: dict,
                   car_dim: tuple) -> TrackResult:
    dna_obj     = Single_DNA_one_chromosome()
    dna_obj.DNA = dna_bits
    nn          = DNA_DECODER(dna_obj)

    clock       = pygame.time.Clock()
    car         = Car(list(track['start']), track['angle'], track['surface'],
                    nn, COMPUTATION, track['collision_map'], car_dim)
    speed_log = []
    timer     = 0

    while car.alive:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        screen.blit(track['surface'], (0, 0))
        car.update(timer)
        speed_log.append(car.speed)
        car.draw(screen)
        pygame.display.set_caption(
            f"{track['name']} | t={timer/FPS:.1f}s | fitness={car.fitness:.1f}"
        )
        pygame.display.update()
        clock.tick(FPS)
        timer += 1

    return TrackResult(track['idx'], car.fitness, speed_log)


# ── phase 1: data collection ──────────────────────────────────────────────────

def collect_data(exp_dirs: list, tracks: list,
                 screen, car_dim: tuple) -> ExperimentData:
    sims = []
    for sim_idx, exp_dir in enumerate(exp_dirs):
        print(f"\n[Sim {sim_idx + 1}/4] {exp_dir}")

        dna_path = os.path.join(exp_dir, "bestValDNA.txt")
        if not os.path.exists(dna_path):
            print(f"  WARNING: {dna_path} not found, skipping.")
            continue
        dna_bits = load_dna(dna_path)

        graph_path = os.path.join(exp_dir, "graphData.txt")
        graph_data = load_graph_data(graph_path) if os.path.exists(graph_path) else []

        track_results = []
        for track in tracks:
            print(f"  {track['name']} ...", end=" ", flush=True)
            result = run_simulation(dna_bits, screen, track, car_dim)
            track_results.append(result)
            print(f"score={result.score:.1f}  {'OK' if result.made_lap else 'FAIL'}")

        sims.append(SimData(sim_idx, exp_dir, graph_data, track_results))

    return ExperimentData(sims, [t['name'] for t in tracks])


# ── metric helpers ────────────────────────────────────────────────────────────

def first_epoch_at_threshold(graph_data: list, col: int, threshold: float):
    for epoch, best, avg, val in graph_data:
        if [epoch, best, avg, val][col] >= threshold:
            return epoch
    return None


def overtraining_ratio(graph_data: list):
    if not graph_data:
        return None
    best_val        = max(val for _, _, _, val in graph_data)
    last_best_epoch = max(epoch for epoch, _, _, val in graph_data if val == best_val)
    total_epochs    = graph_data[-1][0]
    return last_best_epoch / total_epochs if total_epochs else None


# ── phase 2: metrics ──────────────────────────────────────────────────────────
# Each metric: (data: ExperimentData, track_idx: int) -> (float, [detail_str])
# track_idx is a 0-based index into s.track_results for every sim.
# Add/remove entries in METRICS to control what is reported.

def metric_success_rate(data: ExperimentData, track_idx: int):
    """Percentage of sims that completed a lap on this track."""
    results = [s.track_results[track_idx] for s in data.sims]
    n_ok    = sum(1 for r in results if r.made_lap)
    value   = n_ok / len(results) * 100.0 if results else float('nan')
    details = [
        f"    Seed {s.sim_idx+1}: {'OK' if s.track_results[track_idx].made_lap else 'FAIL'}"
        for s in data.sims
    ]
    return value, details


def metric_generalization(data: ExperimentData, track_idx: int):
    """Average test score across all sims on this track."""
    scores  = [s.track_results[track_idx].score for s in data.sims]
    value   = float(np.mean(scores)) if scores else float('nan')
    details = [
        f"    Seed {s.sim_idx+1}: score={s.track_results[track_idx].score:.2f}"
        for s in data.sims
    ]
    return value, details


def metric_stability(data: ExperimentData, track_idx: int):
    """Variance of scores across sims on this track (low = consistent)."""
    scores  = [s.track_results[track_idx].score for s in data.sims]
    value   = float(np.var(scores)) if scores else float('nan')
    details = [
        f"    Seed {s.sim_idx+1}: score={s.track_results[track_idx].score:.2f}"
        for s in data.sims
    ]
    return value, details


def metric_pace_train(data: ExperimentData, track_idx: int):
    """Avg epoch when best_train first reached PACE_THRESHOLD. The same for all tests, max epochs if never did"""
    paces, details = [], []
    for s in data.sims:
        epoch = first_epoch_at_threshold(s.graph_data, col=1, threshold=PACE_THRESHOLD)
        max_e = s.graph_data[-1][0] if s.graph_data else 0
        paces.append(epoch if epoch is not None else max_e)
        details.append(f"    Seed {s.sim_idx+1}: epoch={epoch if epoch is not None else 'never'}")
    return float(np.mean(paces)) if paces else float('nan'), details


def metric_pace_val(data: ExperimentData, track_idx: int):
    """Avg epoch when val_fitness first reached PACE_THRESHOLD. The same for all tests, max epochs if never did"""
    paces, details = [], []
    for s in data.sims:
        epoch = first_epoch_at_threshold(s.graph_data, col=3, threshold=PACE_THRESHOLD)
        max_e = s.graph_data[-1][0] if s.graph_data else 0
        paces.append(epoch if epoch is not None else max_e)
        details.append(f"    Seed {s.sim_idx+1}: epoch={epoch if epoch is not None else 'never'}")
    return float(np.mean(paces)) if paces else float('nan'), details


def metric_success_rate_train(data: ExperimentData, track_idx: int):
    """Percentage of sims where best_train ever reached PACE_THRESHOLD."""
    reached = [
        first_epoch_at_threshold(s.graph_data, col=1, threshold=PACE_THRESHOLD) is not None
        for s in data.sims
    ]
    value   = sum(reached) / len(reached) * 100.0 if reached else float('nan')
    details = [
        f"    Seed {s.sim_idx+1}: {'OK' if r else 'never'}"
        for s, r in zip(data.sims, reached)
    ]
    return value, details


def metric_success_rate_val(data: ExperimentData, track_idx: int):
    """Percentage of sims where val_fitness ever reached PACE_THRESHOLD."""
    reached = [
        first_epoch_at_threshold(s.graph_data, col=3, threshold=PACE_THRESHOLD) is not None
        for s in data.sims
    ]
    value   = sum(reached) / len(reached) * 100.0 if reached else float('nan')
    details = [
        f"    Seed {s.sim_idx+1}: {'OK' if r else 'never'}"
        for s, r in zip(data.sims, reached)
    ]
    return value, details


def metric_quality(data: ExperimentData, track_idx: int):
    """Avg speed variance on this track (lap completers only)."""
    vars_, details = [], []
    for s in data.sims:
        r = s.track_results[track_idx]
        if r.made_lap and r.speed_log:
            v = float(np.var(r.speed_log))
            vars_.append(v)
            details.append(f"    Seed {s.sim_idx+1}: speed_var={v:.5f}")
        else:
            details.append(f"    Seed {s.sim_idx+1}: excluded (no lap)")
    return float(np.mean(vars_)) if vars_ else float('nan'), details


def metric_overtraining(data: ExperimentData, track_idx: int):
    """Avg last-best-val / total epochs. The same for all tests"""
    ratios, details = [], []
    for s in data.sims:

        ratio = overtraining_ratio(s.graph_data)
        if ratio is not None:
            ratios.append(ratio)
            details.append(f"    Seed {s.sim_idx+1}: ratio={ratio:.4f}")

    return float(np.mean(ratios)) if ratios else float('nan'), details


# Registry: (display_label, function, format_string)
METRICS = [
    ("Pace - train         (avg epoch @ 1 lap)                       ", metric_pace_train,         "{:.1f}"),
    ("Pace - val           (avg epoch @ 1 lap)                       ", metric_pace_val,           "{:.1f}"),
    ("Success rate train   (% sims reaching 1 lap in training)       ", metric_success_rate_train, "{:.1f}%"),
    ("Success rate val     (% sims reaching 1 lap in validation)     ", metric_success_rate_val,   "{:.1f}%"),
    ("Success rate test    (% sims completing lap)                   ", metric_success_rate,       "{:.1f}%"),
    ("Generalization       (avg score)                               ", metric_generalization,     "{:.3f}"),
    ("Stability            (variance of scores)                      ", metric_stability,          "{:.3f}"),
    ("Quality              (avg speed var, lap only)                 ", metric_quality,            "{:.5f}"),
    ("Overtraining         (last-best-val / total ep)                ", metric_overtraining,       "{:.4f}"),
]


# ── output ────────────────────────────────────────────────────────────────────

def _fmt(value: float, fmt: str) -> str:
    return fmt.format(value) if not np.isnan(value) else "n/a"


def report(data: ExperimentData, log):
    n = len(data.sims)
    log(f"\n{'='*62}")
    log(f"  EXPERIMENT METRICS  (n={n} sims, {len(data.track_names)} test tracks)")
    log(f"{'='*62}")

    for t_i, t_name in enumerate(data.track_names):
        log(f"\n  - {t_name} -")

        all_details = []
        for label, fn, fmt in METRICS:
            value, details = fn(data, t_i)
            log(f"    {label}: {_fmt(value, fmt)}")
            all_details.append((label.strip(), details))

        log()
        log(f"    Per-seed breakdown [{t_name}]:")
        for label, details in all_details:
            log(f"    [{label}]")
            for line in details:
                log(line)

    log(f"\n{'='*62}")
    log()


def write_csv(data: ExperimentData, path: str):
    # Column names derived from metric function names (strip "metric_" prefix)
    metric_keys = [fn.__name__.replace("metric_", "") for _, fn, _ in METRICS]

    # One row per track; each cell is the plain float value
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["track"] + metric_keys)
        writer.writeheader()
        for t_i, t_name in enumerate(data.track_names):
            row = {"track": t_name}
            for key, (_, fn, _) in zip(metric_keys, METRICS):
                value, _ = fn(data, t_i)
                row[key] = "" if np.isnan(value) else value
            writer.writerow(row)


def combine_results(base_dir: str):
    """
    Scan base_dir for sub-folders that contain test_results.csv,
    merge them into one table with an added 'experiment' column,
    and write it to base_dir/all_results.csv.
    """
    rows = []
    fieldnames = None

    for name in sorted(os.listdir(base_dir)):
        csv_path = os.path.join(base_dir, name, "test_results.csv")
        if not os.path.isfile(csv_path):
            continue
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = ["experiment"] + list(reader.fieldnames)
            for row in reader:
                rows.append({"experiment": name, **row})

    if not rows:
        print(f"No test_results.csv files found under {base_dir}")
        return

    out_path = os.path.join(base_dir, "all_results.csv")
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Combined {len(rows)} rows from {len(set(r['experiment'] for r in rows))} experiments -> {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python test.py <parent_dir>")
        sys.exit(1)

    #parent_dir = sys.argv[1]
    if not os.path.isdir(parent_dir):
        print(f"ERROR: '{parent_dir}' is not a valid directory.")
        sys.exit(1)

    exp_dirs = sorted([
        os.path.join(parent_dir, d)
        for d in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, d))
    ])
    if len(exp_dirs) != 4:
        print(f"ERROR: Expected 4 sub-folders in '{parent_dir}', found {len(exp_dirs)}.")
        sys.exit(1)

    log_path = os.path.join(parent_dir, "test_results.txt")
    log_file = open(log_path, 'w')
    def log(msg=""):
        print(msg)
        log_file.write(msg + "\n")

    pygame.init()
    info    = pygame.display.Info()
    W, H    = int(info.current_w * 0.9), int(info.current_h * 0.9)
    screen  = pygame.display.set_mode((W, H))
    car_dim = (W * 0.01, W * 0.01 * 0.6)

    params = load_params("assets/parameters.txt")
    tracks = load_test_tracks(params, W, H)
    if not tracks:
        log("ERROR: No test tracks found.")
        log("Add assets/test1.png and CAR_TEST1_X/Y/A entries to assets/parameters.txt.")
        log_file.close()
        pygame.quit()
        sys.exit(1)
    print(f"Loaded {len(tracks)} test track(s): {[t['name'] for t in tracks]}")

    data = collect_data(exp_dirs, tracks, screen, car_dim)
    if not data.sims:
        log("ERROR: No simulations completed.")
        log_file.close()
        pygame.quit()
        return

    report(data, log)
    log_file.close()
    print(f"Results saved to {log_path}")

    csv_path = os.path.join(parent_dir, "test_results.csv")
    write_csv(data, csv_path)
    print(f"CSV saved to {csv_path}")
    pygame.quit()


if __name__ == "__main__":
    main()
    #combine_results(R"C:\Users\micha\Desktop\studia\magisterka\Car_evolution\experiements\single_dna\init")