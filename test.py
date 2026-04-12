"""
Test script for evaluating trained car evolution models.

Usage:
    python test.py <parent_dir>

<parent_dir> must contain exactly 4 sub-folders, one per seed.
Each experiment directory must contain:
    - bestValDNA.txt  : DNA string of the best model selected by validation score
    - graphData.txt   : Training history with columns: epoch best_train avg_train val_fitness

Requires in assets/:
    - test_track.png  : Third (test) track image
    - parameters.txt  : Must contain CAR_TEST_X, CAR_TEST_Y, CAR_TEST_A entries

Outputs 4 metrics aggregated across all 4 seeds:
    - Generalization : avg test score across 4 sims
    - Stability      : variance of test scores across 4 sims
    - Pace (train)   : avg epoch when train fitness first reached 300 (3 full laps)
    - Pace (val)     : avg epoch when val fitness first reached 300 (3 full laps)
    - Quality        : avg of per-sim speed variance during test ride (higher = more dynamic)
"""

import sys
import os
import pygame
import numpy as np
from Colors import Color
from Car import Car
from Computation import Computation
from DNA_Decoder import Decoder
from DNA import Single_DNA_one_chromosome


# ── DNA / computation config ─────────────────────────────────────────────────
# Must match the settings used during training (main.py).
INPUTS = [0, 1, 2, 3, 4, 5]
OUTPUTS = [12, 13, 14, 15]
MARKER = [0, 1, 1, 1, 1, 1, 1, 1]

DNA_DECODER = Decoder().decodes_single_DNA_one_chromosome(INPUTS, OUTPUTS, MARKER).fixed_topology
COMPUTATION = Computation(INPUTS, OUTPUTS, MARKER).connection_based_sort_feed_forward

FPS = 1200
PACE_THRESHOLD = 100.0  # fitness points that equal 1 full lap
LAP_THRESHOLD  = 100.0  # minimum test score to count a car as successful


# ── helpers ───────────────────────────────────────────────────────────────────

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
    """Returns list of (epoch, best_train, avg_train, val_fitness)."""
    rows = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                rows.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])))
    return rows


def first_epoch_at_threshold(graph_data: list, col: int, threshold: float):
    """
    Return the first epoch where the value in column `col` >= threshold.
    col: 1 = best_train, 3 = val_fitness
    Returns None if never reached.
    """
    for epoch, best, avg, val in graph_data:
        target = [epoch, best, avg, val][col]
        if target >= threshold:
            return epoch
    return None


def overtraining_ratio(graph_data: list) -> float | None:
    """
    Ratio of the epoch of the last best val fitness to the total number of epochs.
    A value close to 1 means the best val performance arrived very late,
    suggesting the model kept improving (or plateaued) without early stopping —
    a proxy for overtraining / lack of generalisation margin.
    Returns None if graph_data is empty.
    """
    if not graph_data:
        return None
    best_val = max(val for _, _, _, val in graph_data)
    # last epoch that achieved the peak val score
    last_best_epoch = max(epoch for epoch, _, _, val in graph_data if val == best_val)
    total_epochs = graph_data[-1][0]
    if total_epochs == 0:
        return None
    return last_best_epoch / total_epochs


def run_test_simulation(dna_bits, screen, test_track, collision_map,
                        starting_point, starting_angle, car_dim,
                        draw: bool = True) -> tuple:
    """
    Run a single car on the test track.
    Returns (final_fitness, speed_log).
    """
    dna_obj = Single_DNA_one_chromosome()
    dna_obj.DNA = dna_bits
    nn = DNA_DECODER(dna_obj)

    clock = pygame.time.Clock()
    car = Car(list(starting_point), starting_angle, test_track, nn,
              COMPUTATION, collision_map, car_dim)

    speed_log = []
    timer = 0

    while car.alive:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if draw:
            screen.blit(test_track, (0, 0))

        car.update(timer)
        speed_log.append(car.speed)

        if draw:
            car.draw(screen)
            t_str = f"Test sim | time: {timer / FPS:.1f}s | fitness: {car.fitness:.1f}"
            pygame.display.set_caption(t_str)
            pygame.display.update()

        clock.tick(FPS)
        timer += 1

    return car.fitness, speed_log


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python test.py <parent_dir>")
        print("  <parent_dir> must contain exactly 4 sub-folders (one per seed).")
        sys.exit(1)

    parent_dir = sys.argv[1]
    if not os.path.isdir(parent_dir):
        print(f"ERROR: '{parent_dir}' is not a valid directory.")
        sys.exit(1)

    exp_dirs = sorted([
        os.path.join(parent_dir, d)
        for d in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, d))
    ])

    if len(exp_dirs) != 4:
        print(f"ERROR: Expected exactly 4 sub-folders in '{parent_dir}', found {len(exp_dirs)}.")
        sys.exit(1)

    log_path = os.path.join(parent_dir, "test_results.txt")
    log_file = open(log_path, 'w')

    def log(msg=""):
        print(msg)
        log_file.write(msg + "\n")

    # ── pygame setup ─────────────────────────────────────────────────────────
    pygame.init()
    info = pygame.display.Info()
    W = int(info.current_w * 0.9)
    H = int(info.current_h * 0.9)
    screen = pygame.display.set_mode((W, H))

    car_w = W * 0.01
    car_h = car_w * 0.6
    car_dim = (car_w, car_h)

    # ── load test track ───────────────────────────────────────────────────────
    test_track_path = "assets/test_track.png"
    if not os.path.exists(test_track_path):
        log(f"ERROR: {test_track_path} not found. Create the test track first.")
        log_file.close()
        pygame.quit()
        sys.exit(1)

    test_track = pygame.image.load(test_track_path)
    test_track = pygame.transform.scale(test_track, (W, H))

    track_pixels = pygame.surfarray.pixels3d(test_track)
    collision_map = np.all(track_pixels == Color.GREEN, axis=2)
    del track_pixels

    # ── load test starting position ───────────────────────────────────────────
    params = load_params("assets/parameters.txt")
    if "CAR_TEST_X" not in params:
        log("ERROR: CAR_TEST_X / CAR_TEST_Y / CAR_TEST_A not found in assets/parameters.txt.")
        log("Add these three lines to the parameters file for the test track starting position.")
        log_file.close()
        pygame.quit()
        sys.exit(1)

    starting_point = [params["CAR_TEST_X"], params["CAR_TEST_Y"]]
    starting_angle = params["CAR_TEST_A"]

    # ── per-seed collection ───────────────────────────────────────────────────
    test_scores      = []
    speed_variances  = []  # only cars that completed a lap
    train_paces      = []
    val_paces        = []
    overtraining_ratios = []  # only cars that completed a lap
    laps_completed   = []  # bool per sim

    for sim_idx, exp_dir in enumerate(exp_dirs):
        log(f"\n{'='*55}")
        log(f"  Simulation {sim_idx + 1}/4  |  {exp_dir}")
        log(f"{'='*55}")

        # DNA
        dna_path = os.path.join(exp_dir, "bestValDNA.txt")
        if not os.path.exists(dna_path):
            log(f"  WARNING: {dna_path} not found, skipping.")
            continue
        dna_bits = load_dna(dna_path)

        # Training history
        graph_path = os.path.join(exp_dir, "graphData.txt")
        graph_data = load_graph_data(graph_path) if os.path.exists(graph_path) else []

        # Run test sim
        score, speed_log = run_test_simulation(
            dna_bits, screen, test_track, collision_map,
            starting_point, starting_angle, car_dim
        )

        made_lap = score >= LAP_THRESHOLD
        laps_completed.append(made_lap)
        test_scores.append(score)

        # Speed variance only for cars that completed a lap
        if made_lap and speed_log:
            speed_variances.append(float(np.var(speed_log)))

        # Pace of learning from graphData
        train_epoch = first_epoch_at_threshold(graph_data, col=1, threshold=PACE_THRESHOLD)
        val_epoch   = first_epoch_at_threshold(graph_data, col=3, threshold=PACE_THRESHOLD)

        # If threshold was never reached, use total epochs as penalty
        max_epoch = graph_data[-1][0] if graph_data else 0
        train_paces.append(train_epoch if train_epoch is not None else max_epoch)
        val_paces.append(val_epoch   if val_epoch   is not None else max_epoch)

        # Overtraining ratio only for cars that completed a lap
        ot_ratio = overtraining_ratio(graph_data) if made_lap else None
        if ot_ratio is not None:
            overtraining_ratios.append(ot_ratio)

        speed_var_str = f"{float(np.var(speed_log)):.5f}" if (made_lap and speed_log) else "excluded (no lap)"
        ot_str        = f"{ot_ratio:.4f}" if ot_ratio is not None else "excluded (no lap)"
        log(f"  Test score         : {score:.2f}  {'(lap completed)' if made_lap else '(failed lap)'}")
        log(f"  Speed variance     : {speed_var_str}")
        log(f"  Train pace epoch   : {train_epoch if train_epoch is not None else 'never'}")
        log(f"  Val pace epoch     : {val_epoch   if val_epoch   is not None else 'never'}")
        log(f"  Overtraining ratio : {ot_str}")

    # ── aggregate metrics ─────────────────────────────────────────────────────
    n = len(test_scores)
    if n == 0:
        log("\nNo simulations completed successfully.")
        log_file.close()
        pygame.quit()
        return

    n_laps         = sum(laps_completed)
    success_rate   = n_laps / n * 100.0
    generalization = float(np.mean(test_scores))
    stability      = float(np.var(test_scores))
    pace_train     = float(np.mean(train_paces))
    pace_val       = float(np.mean(val_paces))
    quality        = float(np.mean(speed_variances)) if speed_variances else float('nan')
    overtraining   = float(np.mean(overtraining_ratios)) if overtraining_ratios else float('nan')

    log(f"\n{'='*55}")
    log(f"  EXPERIMENT METRICS  (n={n} simulations)")
    log(f"{'='*55}")
    log(f"  Success rate    (cars completing 1 lap)  : {n_laps}/{n}  ({success_rate:.1f}%)")
    log(f"  Generalization  (avg test score)         : {generalization:.3f}")
    log(f"  Stability       (variance test scores)   : {stability:.3f}")
    log(f"  Pace — train    (avg epoch @ 1 lap)      : {pace_train:.1f}")
    log(f"  Pace — val      (avg epoch @ 1 lap)      : {pace_val:.1f}")
    log(f"  Quality         (avg speed var, lap only): {quality:.5f}")
    log(f"  Overtraining    (avg last-best-val/total): {overtraining:.4f}")
    log(f"{'='*55}")
    log()

    # Raw values per seed for reference
    log("  Per-seed details:")
    sv_iter = iter(speed_variances)
    ot_iter = iter(overtraining_ratios)
    for i in range(n):
        made = laps_completed[i]
        sv   = next(sv_iter) if made else None
        ot   = next(ot_iter) if made else None
        sv_s = f"{sv:.5f}" if sv is not None else "n/a"
        ot_s = f"{ot:.4f}" if ot is not None else "n/a"
        log(f"    Seed {i+1}: score={test_scores[i]:.2f}  "
            f"speed_var={sv_s}  "
            f"train_pace={train_paces[i]}  "
            f"val_pace={val_paces[i]}  "
            f"overtraining={ot_s}")
    log()

    log_file.close()
    print(f"Results saved to {log_path}")
    pygame.quit()


if __name__ == "__main__":
    main()