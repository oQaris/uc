"""
ML-Assisted Unit Commitment Solver

Based on: Ramesh & Li, "Machine Learning Assisted Model Reduction for Security-Constrained Unit Commitment", NAPS 2022.
https://arxiv.org/abs/2111.09824

Approach:
1. Train logistic regression models on historical demand → commitment data
2. For a new instance, predict generator commitment status
3. Fix predicted always-ON / always-OFF generators in the Pyomo model
4. Solve the reduced MILP (fewer binary variables → faster)
"""

import json
import os
import pickle
import time

import numpy as np
from pyomo.environ import TerminationCondition
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from src.models.uc_model import build_uc_model


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(data):
    """Extract feature vector from a UC instance.

    Features (all normalized by peak demand):
    - demand profile (T values)
    - reserves profile (T values)
    """
    demand = np.array(data["demand"], dtype=np.float64)
    reserves = np.array(data["reserves"], dtype=np.float64)
    peak = demand.max()
    if peak == 0:
        peak = 1.0
    return np.concatenate([demand / peak, reserves / peak])


# ---------------------------------------------------------------------------
# Training data generation
# ---------------------------------------------------------------------------

def solve_instance_for_labels(data, solver_name="appsi_highs", gap=0.01,
                              time_limit=300, threads=1,
                              first_feasible=False):
    """Solve a UC instance and return commitment decisions.

    Parameters
    ----------
    first_feasible : bool
        If True, stop as soon as the first feasible (integer) solution is
        found.  This is much faster for hard instances and still produces
        useful training labels — the ML model only needs to learn which
        generators are consistently ON/OFF across samples.

    Returns
    -------
    commitment : dict  {gen_name: [ug_1, ..., ug_T]}
    objective  : float
    """
    from pyomo.opt import SolverFactory
    from pyomo.environ import value as pyo_value

    model = build_uc_model(data)
    solver = SolverFactory(solver_name)

    if solver_name == "appsi_highs":
        solver.options["threads"] = threads
        solver.options["mip_rel_gap"] = gap
        if first_feasible:
            solver.options["solution_limit"] = 1
        if time_limit:
            solver.options["time_limit"] = float(time_limit)
    elif solver_name == "cbc":
        solver.options["ratioGap"] = gap
        solver.options["threads"] = threads
        if first_feasible:
            solver.options["maxSolutions"] = 1
        if time_limit:
            solver.options["seconds"] = time_limit

    result = solver.solve(model, tee=False)

    # Check feasibility
    ok_conditions = {TerminationCondition.optimal, TerminationCondition.feasible}
    if result.solver.termination_condition not in ok_conditions:
        raise RuntimeError(
            f"Solver returned {result.solver.termination_condition}")

    T = data["time_periods"]
    commitment = {}
    for g in data["thermal_generators"]:
        commitment[g] = [
            int(round(pyo_value(model.ug[g, t])))
            for t in range(1, T + 1)
        ]

    objective = float(pyo_value(model.obj))
    return commitment, objective


def augment_instance(data, scale_factor=1.0, noise_std=0.03, seed=None):
    """Create an augmented copy of a UC instance.

    Demand and reserves are scaled by *scale_factor* and then perturbed with
    multiplicative Gaussian noise  ``d' = d * scale * (1 + N(0, noise_std))``.
    Negative values are clipped to 0.
    """
    import copy
    rng = np.random.RandomState(seed)
    aug = copy.deepcopy(data)

    demand = np.array(data["demand"], dtype=np.float64)
    reserves = np.array(data["reserves"], dtype=np.float64)

    noise_d = 1.0 + rng.normal(0, noise_std, size=demand.shape)
    noise_r = 1.0 + rng.normal(0, noise_std, size=reserves.shape)

    aug["demand"] = np.maximum(demand * scale_factor * noise_d, 0).tolist()
    aug["reserves"] = np.maximum(reserves * scale_factor * noise_r, 0).tolist()
    return aug


def _cache_path_for(cache_dir, fname, tag):
    """Build cache file path for an (instance, augmentation) pair."""
    key = f"{fname.replace('.json', '')}_{tag}"
    return os.path.join(cache_dir, f"{key}.npz")


def _remap_labels(cached_gen_names, cached_labels, union_gen_names, T):
    """Remap cached labels from instance gen order to union gen order.

    Generators present in *union_gen_names* but absent from
    *cached_gen_names* get NaN labels.
    """
    result = np.full(len(union_gen_names) * T, np.nan)
    cached_idx = {g: i for i, g in enumerate(cached_gen_names)}
    for ui, g in enumerate(union_gen_names):
        if g in cached_idx:
            ci = cached_idx[g]
            result[ui * T: ui * T + T] = cached_labels[ci * T: ci * T + T]
    return result


def generate_training_data(data_dir, solver_name="appsi_highs", gap=0.02,
                           time_limit=300, threads=1,
                           n_augmented=5, noise_std=0.03,
                           scale_range=(0.9, 1.1), cache_dir=None,
                           first_feasible=False, verbose=True):
    """Generate training dataset from UC instances in *data_dir*.

    Supports datasets where different instances have different generator sets.
    The union of all generator names is used; labels for absent generators
    are set to NaN.

    Returns
    -------
    X : ndarray (n_samples, n_features)
    Y : ndarray (n_samples, n_generators * T)  — may contain NaN
    gen_names : list[str]   — union of generators across all instances
    T : int
    """
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {data_dir}")

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    # First pass: collect union of all generator names and T
    all_gen_names = set()
    T = None
    for fname in files:
        fpath = os.path.join(data_dir, fname)
        with open(fpath) as fp:
            data = json.load(fp)
        all_gen_names.update(data["thermal_generators"].keys())
        if T is None:
            T = data["time_periods"]
    gen_names = sorted(all_gen_names)

    if verbose:
        print(f"Union of generators: {len(gen_names)} across {len(files)} instances")

    # Second pass: solve instances and build training data
    X_list, Y_list = [], []
    n_cached = 0

    for fi, fname in enumerate(files):
        fpath = os.path.join(data_dir, fname)
        with open(fpath) as fp:
            data = json.load(fp)

        inst_gen_names = sorted(data["thermal_generators"].keys())

        instances_to_solve = [("original", data)]
        rng = np.random.RandomState(hash(fname) % (2 ** 31))
        for ai in range(n_augmented):
            scale = rng.uniform(*scale_range)
            aug = augment_instance(data, scale_factor=scale,
                                   noise_std=noise_std, seed=rng.randint(2 ** 31))
            instances_to_solve.append((f"aug-{ai}", aug))

        for tag, inst_data in instances_to_solve:
            label = f"[{fi + 1}/{len(files)}] {fname} ({tag})"

            # Try loading from cache (only labels; features are re-extracted)
            if cache_dir:
                cp = _cache_path_for(cache_dir, fname, tag)
                if os.path.exists(cp):
                    cached = np.load(cp, allow_pickle=True)
                    if "gen_names" in cached:
                        cached_gn = list(cached["gen_names"])
                        labels = _remap_labels(
                            cached_gn, cached["labels"], gen_names, T)
                        X_list.append(extract_features(inst_data))
                        Y_list.append(labels)
                        n_cached += 1
                        if verbose:
                            print(f"  {label} ... CACHED")
                        continue
                    else:
                        # Old cache format without gen_names — delete and re-solve
                        if verbose:
                            print(f"  {label} ... old cache format, re-solving")
                        os.remove(cp)

            if verbose:
                print(f"  Solving {label} ...", end=" ", flush=True)
            t0 = time.time()
            try:
                commitment, obj = solve_instance_for_labels(
                    inst_data, solver_name, gap, time_limit, threads,
                    first_feasible=first_feasible)
            except Exception as e:
                if verbose:
                    print(f"FAILED: {e}")
                continue
            elapsed = time.time() - t0
            if verbose:
                print(f"OK  obj={obj:.0f}  time={elapsed:.1f}s")

            features = extract_features(inst_data)

            # Build labels in union gen order; NaN for missing generators
            labels = np.full(len(gen_names) * T, np.nan)
            for gi, g in enumerate(gen_names):
                if g in commitment:
                    for t in range(T):
                        labels[gi * T + t] = commitment[g][t]

            X_list.append(features)
            Y_list.append(labels)

            # Save to cache with instance gen_names for remapping on load
            if cache_dir:
                inst_labels = []
                for g in inst_gen_names:
                    inst_labels.extend(commitment[g])
                np.savez(cp, features=features,
                         labels=np.array(inst_labels),
                         gen_names=np.array(inst_gen_names))

    X = np.array(X_list)
    Y = np.array(Y_list)
    if verbose:
        cached_msg = f" ({n_cached} from cache)" if n_cached else ""
        n_nan_cols = np.isnan(Y).any(axis=0).sum()
        nan_msg = f", {n_nan_cols} columns with NaN" if n_nan_cols else ""
        print(f"\nTraining data: {X.shape[0]} samples{cached_msg}, "
              f"{X.shape[1]} features, {Y.shape[1]} labels "
              f"({len(gen_names)} gens x {T} periods{nan_msg})")
    return X, Y, gen_names, T


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def _create_classifier(method, random_state=0):
    """Create a classifier instance for the given method.

    Parameters
    ----------
    method : str
        One of ``"lr"`` (logistic regression), ``"rf"`` (random forest),
        ``"cb"`` (CatBoost gradient boosting).

    Returns
    -------
    classifier : sklearn-compatible estimator
    tag : str
    """
    if method == "lr":
        return LogisticRegression(
            random_state=random_state, solver="liblinear",
            max_iter=300, C=1.0,
        ), "lr"
    elif method == "rf":
        return RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=random_state,
            n_jobs=1,
        ), "rf"
    elif method == "cb":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=200, depth=4, learning_rate=0.1,
            random_seed=random_state, verbose=False,
            auto_class_weights="Balanced",
            thread_count=1,
        ), "cb"
    elif method == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=150,
            random_state=random_state,
        ), "mlp"
    elif method == "svm":
        return SVC(
            kernel="rbf", C=1.0, gamma="scale",
            probability=True,
            random_state=random_state,
        ), "svm"
    elif method == "knn":
        return KNeighborsClassifier(
            n_neighbors=3, weights="distance",
        ), "knn"
    else:
        raise ValueError(f"Unknown ML method: {method!r}. "
                         f"Use 'lr', 'rf', 'cb', 'mlp', 'svm', or 'knn'.")


def train_commitment_models(X, Y, gen_names, T, method="lr", verbose=True):
    """Train one classifier per (generator, period).

    Parameters
    ----------
    X : ndarray (n_samples, n_features)
    Y : ndarray (n_samples, n_generators * T) — may contain NaN
    gen_names : list[str]
    T : int
    method : str
        ``"lr"`` — logistic regression (default),
        ``"rf"`` — random forest,
        ``"cb"`` — CatBoost gradient boosting.

    NaN values in Y are filtered out per-column: only samples where the
    generator was present contribute to training.  If a generator has no
    valid samples at all, it gets ``"no_model"`` tags and is skipped
    during prediction (left free for the solver).

    Returns
    -------
    models_dict : dict  {gen_name: [(model_or_int, tag), ...] * T}
    scores : list of float
        Training accuracy for columns that were actually trained.
    """
    METHOD_NAMES = {"lr": "LR", "rf": "RF", "cb": "CatBoost",
                    "mlp": "MLP", "svm": "SVM", "knn": "KNN"}

    models_dict = {}
    scores = []
    n_trained = 0
    n_const = 0
    n_no_model = 0

    for gi, g in enumerate(gen_names):
        period_models = []
        has_any_model = False
        for t in range(T):
            col_idx = gi * T + t
            y_col = Y[:, col_idx]
            mask = ~np.isnan(y_col)

            if mask.sum() == 0:
                period_models.append((None, "no_model"))
                n_no_model += 1
                continue

            y_valid = y_col[mask].astype(int)
            X_valid = X[mask]
            unique = np.unique(y_valid)

            if len(unique) == 1:
                val = int(unique[0])
                period_models.append((val, f"const_{val}"))
                scores.append(1.0)
                n_const += 1
                has_any_model = True
            else:
                clf, tag = _create_classifier(method)
                # KNN: n_neighbors не может превышать число сэмплов
                if method == "knn" and len(X_valid) < clf.n_neighbors:
                    clf.n_neighbors = max(1, len(X_valid))
                clf.fit(X_valid, y_valid)
                sc = clf.score(X_valid, y_valid)
                period_models.append((clf, tag))
                scores.append(sc)
                n_trained += 1
                has_any_model = True

        if has_any_model:
            models_dict[g] = period_models

    if verbose:
        mean_sc = np.mean(scores) if scores else 0
        label = METHOD_NAMES.get(method, method)
        parts = [f"{n_trained} {label} models", f"{n_const} constant"]
        if n_no_model:
            parts.append(f"{n_no_model} no_model")
        print(f"Trained {', '.join(parts)}, "
              f"mean accuracy = {mean_sc:.4f}")

    return models_dict, scores


def save_models(models_dict, T, path):
    """Persist trained models and metadata to disk."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"models_dict": models_dict, "T": T}, f)


def load_models(path):
    """Load previously saved models.

    Returns (models_dict, T).
    """
    with open(path, "rb") as f:
        d = pickle.load(f)
    if "models_dict" not in d:
        raise ValueError(
            f"Old model format in {path}. Delete the file and retrain.")
    return d["models_dict"], d["T"]


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_commitment(models_dict, features, T):
    """Predict commitment for every (generator, period).

    Only generators present in *models_dict* (and not all-no_model) are
    returned.  Generators absent from the result should be left free
    for the solver.

    Returns dict  {gen_name: [pred_1, ..., pred_T]}.
    """
    x = features.reshape(1, -1)
    predictions = {}
    for g, period_models in models_dict.items():
        # Skip generators with no usable models
        if all(tag == "no_model" for _, tag in period_models):
            continue
        preds = []
        for model, mtype in period_models:
            if mtype.startswith("const_"):
                preds.append(int(mtype[-1]))
            elif mtype == "no_model":
                # No data for this period; default to 0 (will likely make
                # the generator "mixed" and thus free for the solver)
                preds.append(0)
            else:
                preds.append(int(model.predict(x)[0]))
        predictions[g] = preds
    return predictions


def predict_commitment_proba(models_dict, features, T):
    """Predict commitment probabilities for every (generator, period).

    Returns dict  {gen_name: [proba_1, ..., proba_T]} where proba_t is the
    probability that the generator is ON at period t.
    """
    x = features.reshape(1, -1)
    probas = {}
    for g, period_models in models_dict.items():
        if all(tag == "no_model" for _, tag in period_models):
            continue
        proba_list = []
        for model, mtype in period_models:
            if mtype == "const_0":
                proba_list.append(0.0)
            elif mtype == "const_1":
                proba_list.append(1.0)
            elif mtype == "no_model":
                proba_list.append(0.5)  # unknown → uncertain
            else:
                proba_list.append(float(model.predict_proba(x)[0, 1]))
        probas[g] = proba_list
    return probas


# ---------------------------------------------------------------------------
# Variable fixing (MILP reduction)
# ---------------------------------------------------------------------------

def _check_initial_feasibility(gen_data, fix_value, T):
    """Check if fixing a generator to *fix_value* for all periods is
    compatible with its initial conditions.

    Returns the number of leading periods that MUST keep their initial value
    (due to min-up / min-down constraints from t0).
    """
    if fix_value == 0 and gen_data["unit_on_t0"] == 1:
        # Generator was ON at t0 — must stay ON for remaining min-uptime
        remaining = gen_data["time_up_minimum"] - gen_data["time_up_t0"]
        if remaining > 0:
            return min(remaining, T)
    elif fix_value == 1 and gen_data["unit_on_t0"] == 0:
        # Generator was OFF at t0 — must stay OFF for remaining min-downtime
        remaining = gen_data["time_down_minimum"] - gen_data["time_down_t0"]
        if remaining > 0:
            return min(remaining, T)
    return 0


def _feasibility_check(predictions, data, verbose=True):
    """Feasibility layer: ensure predicted commitment is physically feasible.

    Two checks per time period:
    1. **Capacity check**: max generation of ON+free generators >= demand
       If not, convert cheapest predicted-OFF generators to free.
    2. **Min-generation check**: total pmin of always-ON generators <= demand
       If not, convert most-expensive always-ON generators to free.

    Both checks include a 10% reserve margin for robustness.

    Generators absent from *predictions* are implicitly free and always
    contribute capacity.

    Modifies *predictions* in-place and returns the number of generators
    whose status was changed.
    """
    T = data["time_periods"]
    thermal = data["thermal_generators"]
    gen_names = sorted(thermal.keys())
    MARGIN = 1.10  # 10% reserve margin

    # Only classify generators that have predictions
    always_off = set(
        g for g in gen_names
        if g in predictions and all(p == 0 for p in predictions[g]))
    always_on = set(
        g for g in gen_names
        if g in predictions and all(p == 1 for p in predictions[g]))
    changed = set()

    # --- Check 1: enough max capacity to meet demand ---
    for t_idx in range(T):
        demand_t = data["demand"][t_idx] + data["reserves"][t_idx]
        # Capacity from non-OFF generators (ON + free + no-prediction)
        active = [g for g in gen_names if g not in always_off or g in changed]
        cap = sum(thermal[g]["power_output_maximum"] for g in active)

        if cap >= demand_t * MARGIN:
            continue

        # Rescue cheapest OFF generators
        off_candidates = sorted(
            [g for g in always_off if g not in changed],
            key=lambda g: thermal[g]["piecewise_production"][0]["cost"]
        )
        for g in off_candidates:
            changed.add(g)
            cap += thermal[g]["power_output_maximum"]
            if cap >= demand_t * MARGIN:
                break

    # --- Check 2: min generation of always-ON doesn't exceed demand ---
    for t_idx in range(T):
        demand_t = data["demand"][t_idx]
        # Total minimum output of generators forced ON
        on_gens = [g for g in always_on if g not in changed]
        min_gen = sum(thermal[g]["power_output_minimum"] for g in on_gens)

        if min_gen <= demand_t:
            continue

        # Too much forced generation — free the most expensive ON generators
        on_sorted = sorted(
            on_gens,
            key=lambda g: thermal[g]["piecewise_production"][-1]["cost"],
            reverse=True,  # most expensive first
        )
        for g in on_sorted:
            changed.add(g)
            min_gen -= thermal[g]["power_output_minimum"]
            if min_gen <= demand_t:
                break

    # Convert changed generators to "mixed" (leave for solver)
    for g in changed:
        # Alternate predictions so the generator won't be classified as
        # always-on or always-off → it stays free in apply_ml_reduction.
        predictions[g] = [0, 1] * (T // 2) + [0] * (T % 2)

    if verbose and changed:
        n_from_off = len(changed & always_off)
        n_from_on = len(changed & always_on)
        parts = []
        if n_from_off:
            parts.append(f"{n_from_off} from OFF")
        if n_from_on:
            parts.append(f"{n_from_on} from ON")
        print(f"  Feasibility layer: freed {len(changed)} generators "
              f"({', '.join(parts)})")

    return len(changed)


def apply_ml_reduction(pyomo_model, predictions, data, confidence_threshold=None,
                       probabilities=None, verbose=True):
    """Fix variables in the Pyomo model based on ML predictions.

    Reduction strategy (from paper):
    - Generator predicted OFF for ALL periods → fix ug[g,t]=0 ∀t
    - Generator predicted ON for ALL periods  → fix ug[g,t]=1 ∀t
    - Mixed predictions → leave as binary (solver decides)
    - Generator not in predictions → leave as binary (solver decides)

    Returns
    -------
    stats : dict with keys n_fixed_off, n_fixed_on, n_free,
            n_vars_fixed, n_vars_total
    """
    T = data["time_periods"]
    thermal = data["thermal_generators"]
    gen_names = sorted(thermal.keys())

    n_fixed_off = 0
    n_fixed_on = 0
    n_free = 0
    n_vars_fixed = 0

    for g in gen_names:
        # Generator without prediction → free for solver
        if g not in predictions:
            n_free += 1
            continue

        pred = predictions[g]
        gen = thermal[g]

        # Optional confidence filter
        if confidence_threshold is not None and probabilities is not None:
            if g not in probabilities:
                n_free += 1
                continue
            probs = probabilities[g]
            all_off = all(p < (1 - confidence_threshold) for p in probs)
            all_on = all(p > confidence_threshold for p in probs)
        else:
            all_off = all(p == 0 for p in pred)
            all_on = all(p == 1 for p in pred)

        if all_off:
            blocked = _check_initial_feasibility(gen, 0, T)
            if blocked >= T:
                n_free += 1
                continue
            t_start = blocked + 1  # first period we can fix
            for t in range(t_start, T + 1):
                pyomo_model.ug[g, t].fix(0)
                n_vars_fixed += 1
                # At the boundary period, leave vg/wg free so the
                # logical constraint ug[t] - ug[t-1] = vg[t] - wg[t]
                # can be satisfied (may need wg=1 for shutdown).
                if t > t_start or (t == 1 and gen["unit_on_t0"] == 0):
                    pyomo_model.vg[g, t].fix(0)
                    pyomo_model.wg[g, t].fix(0)
                    n_vars_fixed += 2
            n_fixed_off += 1

        elif all_on:
            blocked = _check_initial_feasibility(gen, 1, T)
            if blocked >= T:
                n_free += 1
                continue
            t_start = blocked + 1
            for t in range(t_start, T + 1):
                pyomo_model.ug[g, t].fix(1)
                n_vars_fixed += 1
                pyomo_model.wg[g, t].fix(0)  # no shutdown
                n_vars_fixed += 1
                # At the boundary period, leave vg free (may need
                # startup if previous period was OFF).
                if t > t_start or (t == 1 and gen["unit_on_t0"] == 1):
                    pyomo_model.vg[g, t].fix(0)
                    n_vars_fixed += 1
            n_fixed_on += 1

        else:
            n_free += 1

    n_total_binary = len(gen_names) * T * 3  # ug + vg + wg per (g,t)
    if verbose:
        print(f"ML reduction: {n_fixed_off} gens fixed OFF, "
              f"{n_fixed_on} gens fixed ON, {n_free} gens free")
        print(f"  Binary vars fixed: {n_vars_fixed} / {n_total_binary} "
              f"({100 * n_vars_fixed / max(n_total_binary, 1):.1f}%)")

    return {
        "n_fixed_off": n_fixed_off,
        "n_fixed_on": n_fixed_on,
        "n_free": n_free,
        "n_vars_fixed": n_vars_fixed,
        "n_vars_total": n_total_binary,
    }


# ---------------------------------------------------------------------------
# Full ML-assisted solving pipeline
# ---------------------------------------------------------------------------

def solve_ml_assisted(data, models_dict, T,
                      solver_name="appsi_highs", gap=0.01,
                      time_limit=None, threads=1,
                      confidence_threshold=None, verbose=True):
    """Solve a UC instance using ML-assisted MILP reduction.

    Pipeline:
    1. Extract features from *data*
    2. Predict commitment with trained models
    3. Build Pyomo model
    4. Fix variables based on predictions
    5. Solve reduced MILP

    Parameters
    ----------
    models_dict : dict  {gen_name: [(model, tag), ...] * T}
        Trained models as returned by ``train_commitment_models``.
        Generators present in the test instance but absent from
        *models_dict* are left free for the solver.

    Returns
    -------
    result : dict with keys
        objective, solve_time, build_time, status, reduction_stats,
        predictions
    """
    from pyomo.opt import SolverFactory
    from pyomo.environ import value as pyo_value, TerminationCondition

    # 1. Feature extraction
    features = extract_features(data)

    # 2. Prediction
    predictions = predict_commitment(models_dict, features, T)
    probabilities = None
    if confidence_threshold is not None:
        probabilities = predict_commitment_proba(models_dict, features, T)

    # 2b. Feasibility layer — ensure enough capacity to meet demand
    _feasibility_check(predictions, data, verbose=verbose)

    # 3. Build model
    t0 = time.time()
    pyomo_model = build_uc_model(data)
    build_time = time.time() - t0

    # 4. Apply ML reduction
    reduction_stats = apply_ml_reduction(
        pyomo_model, predictions, data,
        confidence_threshold=confidence_threshold,
        probabilities=probabilities,
        verbose=verbose,
    )

    # 5. Solve reduced model
    def _make_solver():
        s = SolverFactory(solver_name)
        if solver_name == "appsi_highs":
            s.options["threads"] = threads
            s.options["mip_rel_gap"] = gap
            if time_limit:
                s.options["time_limit"] = float(time_limit)
        elif solver_name == "cbc":
            s.options["ratioGap"] = gap
            s.options["threads"] = threads
            if time_limit:
                s.options["seconds"] = time_limit
        return s

    solver = _make_solver()

    t0 = time.time()
    feasible = True
    try:
        solve_result = solver.solve(pyomo_model, tee=False,
                                    load_solutions=False)
        # Check if we got a feasible solution
        if solve_result.solver.termination_condition in {
            TerminationCondition.optimal, TerminationCondition.feasible}:
            pyomo_model.solutions.load_from(solve_result)
        else:
            feasible = False
    except (RuntimeError, Exception):
        feasible = False

    if not feasible:
        # Fallback: rebuild model without ML reductions.
        if verbose:
            print("  Reduced model infeasible — falling back to standard solve")
        pyomo_model = build_uc_model(data)
        solver = _make_solver()
        solve_result = solver.solve(pyomo_model, tee=False)
        n_gens = len(data["thermal_generators"])
        reduction_stats = {
            "n_fixed_off": 0, "n_fixed_on": 0,
            "n_free": n_gens,
            "n_vars_fixed": 0,
            "n_vars_total": reduction_stats["n_vars_total"],
        }
    solve_time = time.time() - t0

    # Parse status
    ok_conditions = {TerminationCondition.optimal, TerminationCondition.feasible}
    if solve_result.solver.termination_condition in ok_conditions:
        status = str(solve_result.solver.termination_condition)
        objective = float(pyo_value(pyomo_model.obj))
    else:
        status = str(solve_result.solver.termination_condition)
        try:
            objective = float(pyo_value(pyomo_model.obj))
        except:
            objective = None

    if verbose:
        obj_str = f"{objective:.0f}" if objective else "N/A"
        print(f"ML-assisted solve: status={status}, obj={obj_str}, "
              f"solve_time={solve_time:.2f}s, build_time={build_time:.2f}s")

    return {
        "objective": objective,
        "solve_time": solve_time,
        "build_time": build_time,
        "total_time": build_time + solve_time,
        "status": status,
        "reduction_stats": reduction_stats,
        "predictions": predictions,
    }


def solve_standard(data, solver_name="appsi_highs", gap=0.01,
                   time_limit=None, threads=1, verbose=True):
    """Solve a UC instance with standard MILP (no ML reduction).

    Used as baseline for comparison with ML-assisted approach.
    """
    from pyomo.opt import SolverFactory
    from pyomo.environ import value as pyo_value, TerminationCondition

    t0 = time.time()
    pyomo_model = build_uc_model(data)
    build_time = time.time() - t0

    solver = SolverFactory(solver_name)
    if solver_name == "appsi_highs":
        solver.options["threads"] = threads
        solver.options["mip_rel_gap"] = gap
        if time_limit:
            solver.options["time_limit"] = float(time_limit)
    elif solver_name == "cbc":
        solver.options["ratioGap"] = gap
        solver.options["threads"] = threads
        if time_limit:
            solver.options["seconds"] = time_limit

    t0 = time.time()
    solve_result = solver.solve(pyomo_model, tee=verbose and False)
    solve_time = time.time() - t0

    ok_conditions = {TerminationCondition.optimal, TerminationCondition.feasible}
    if solve_result.solver.termination_condition in ok_conditions:
        status = str(solve_result.solver.termination_condition)
        objective = float(pyo_value(pyomo_model.obj))
    else:
        status = str(solve_result.solver.termination_condition)
        try:
            objective = float(pyo_value(pyomo_model.obj))
        except:
            objective = None

    if verbose:
        obj_str = f"{objective:.0f}" if objective else "N/A"
        print(f"Standard solve: status={status}, obj={obj_str}, "
              f"solve_time={solve_time:.2f}s, build_time={build_time:.2f}s")

    return {
        "objective": objective,
        "solve_time": solve_time,
        "build_time": build_time,
        "total_time": build_time + solve_time,
        "status": status,
    }
