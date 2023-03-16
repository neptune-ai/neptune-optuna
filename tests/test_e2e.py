import deepdiff
import optuna
import pytest

try:
    from neptune import init_run
except ImportError:
    from neptune.new import init_run

import neptune_optuna.impl as npt_utils


@pytest.mark.parametrize("handler_namespace", [None, "handler_namespace"])
@pytest.mark.parametrize("base_namespace", ["", "base_namespace"])
def test_callback(handler_namespace, base_namespace):

    run = init_run()

    if handler_namespace is not None:
        handler = run[handler_namespace]
    else:
        handler = run

    neptune_callback = npt_utils.NeptuneCallback(handler, base_namespace=base_namespace)

    def objective(trial):
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_float("y", -10, 10)
        return (x + y) ** 2

    n_trials = 5
    study = optuna.create_study()
    study.optimize(objective, n_trials=n_trials, callbacks=[neptune_callback])

    validate_run(run, n_trials, study, handler_namespace, base_namespace)
    assert run["source_code/integrations/neptune-optuna"].fetch() == npt_utils.__version__

    run.stop()


def test_log_and_load_study():
    def objective(trial):
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_float("y", -10, 10)
        return (x + y) ** 2

    n_trials = 5
    study = optuna.create_study()
    study.optimize(objective, n_trials=n_trials)

    run = init_run()
    npt_utils.log_study_metadata(study, run)

    validate_run(run, n_trials, study)

    # Loaded study is the same as the saved one
    validate_loaded_study(run, study)

    run.stop()


def _prefix(handler_namespace, base_namespace):
    prefix = ""
    if handler_namespace is not None:
        prefix = f"{handler_namespace}/"
    if base_namespace != "":
        prefix = f"{prefix}{base_namespace}/"
    return prefix


def validate_loaded_study(run, study):
    run.wait()
    loaded_study = npt_utils.load_study_from_run(run)
    assert isinstance(loaded_study, optuna.study.Study)
    assert deepdiff.DeepDiff(loaded_study, study) == {}


def validate_run(run, n_trials, study, handler_namespace=None, base_namespace=""):
    run.wait()
    prefix = _prefix(handler_namespace, base_namespace)

    assert run.exists(f"{prefix}best")
    assert run.exists(f"{prefix}study")
    assert run.exists(f"{prefix}trials")
    assert run.exists(f"{prefix}visualizations")

    run_structure = run.get_structure()
    if handler_namespace is not None:
        run_structure = run_structure[handler_namespace]
    if base_namespace != "":
        run_structure = run_structure[base_namespace]
    assert len(run_structure["trials"]["trials"]) == n_trials

    assert len(run[f"{prefix}trials/values"].fetch_values()) == n_trials
    assert len(run[f"{prefix}trials/params/x"].fetch_values()) == n_trials

    assert run[f"{prefix}best/params"].fetch() == study.best_params

    assert run.exists(f"{prefix}study/study_name")
    assert run.exists(f"{prefix}study/distributions/")
