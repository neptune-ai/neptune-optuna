import deepdiff
import optuna
import pytest

try:
    from neptune import init_run
except ImportError:
    from neptune.new import init_run

from neptune.utils import stringify_unsupported

import neptune_optuna.impl as npt_utils

dummy_user_attr = [1, "a"]


@pytest.mark.parametrize("log_all_trials", [True, False])
@pytest.mark.parametrize("handler_namespace", [None, "handler_namespace"])
@pytest.mark.parametrize("base_namespace", ["", "base_namespace"])
@pytest.mark.parametrize(
    "prune_condition",
    [
        lambda trial: False,
        lambda trial: trial.number == 0,
    ],
)
def test_callback(handler_namespace, base_namespace, log_all_trials, prune_condition):
    run = init_run()

    handler = run[handler_namespace] if handler_namespace is not None else run
    neptune_callback = npt_utils.NeptuneCallback(handler, base_namespace=base_namespace, log_all_trials=log_all_trials)

    def objective(trial):
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_float("y", -10, 10)
        trial.set_user_attr("dummy_trial_key", dummy_user_attr)
        if prune_condition(trial):
            raise optuna.TrialPruned()
        return (x + y) ** 2

    n_trials = 5
    study = optuna.create_study()
    study.set_user_attr("dummy_study_key", dummy_user_attr)
    study.optimize(objective, n_trials=n_trials, callbacks=[neptune_callback])

    validate_run(run, n_trials, study, handler_namespace, base_namespace, log_all_trials)

    if log_all_trials:
        prefix = _prefix(handler_namespace, base_namespace)
        for i in range(n_trials):
            trial = study.trials[i]
            is_pruned = trial.state == optuna.trial.TrialState.PRUNED
            if i == 0:
                assert run[f"{prefix}trials/trials/{i}/is_pruned"].fetch() == is_pruned
            else:
                assert not run[f"{prefix}trials/trials/{i}/is_pruned"].fetch()

    assert run["source_code/integrations/neptune-optuna"].fetch() == npt_utils.__version__

    run.stop()


def test_log_and_load_study():
    def objective(trial):
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_float("y", -10, 10)
        trial.set_user_attr("dummy_trial_key", dummy_user_attr)
        return (x + y) ** 2

    n_trials = 5
    study = optuna.create_study()
    study.set_user_attr("dummy_study_key", dummy_user_attr)
    study.optimize(objective, n_trials=n_trials)

    run = init_run()
    npt_utils.log_study_metadata(study, run)

    validate_run(run, n_trials, study)

    # Loaded study is the same as the saved one
    validate_loaded_study(run, study)

    run.stop()


def _prefix(handler_namespace, base_namespace):
    prefix = f"{handler_namespace}/" if handler_namespace is not None else ""
    if base_namespace != "":
        prefix = f"{prefix}{base_namespace}/"
    return prefix


def validate_loaded_study(run, study):
    run.wait()
    loaded_study = npt_utils.load_study_from_run(run)
    assert isinstance(loaded_study, optuna.study.Study)
    assert deepdiff.DeepDiff(loaded_study, study) == {}


def validate_run(run, n_trials, study, handler_namespace=None, base_namespace="", log_all_trials=True):
    run.wait()
    prefix = _prefix(handler_namespace, base_namespace)

    assert run.exists(f"{prefix}best")
    assert run.exists(f"{prefix}study")
    assert run.exists(f"{prefix}visualizations")

    run_structure = run.get_structure()
    if handler_namespace is not None:
        run_structure = run_structure[handler_namespace]
    if base_namespace != "":
        run_structure = run_structure[base_namespace]

    if log_all_trials:
        assert run.exists(f"{prefix}trials")

        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        n_completed = len(completed_trials)

        assert len(run_structure["trials"]["trials"]) == n_trials

        if n_completed > 0:
            assert len(run[f"{prefix}trials/values"].fetch_values()) == n_completed
            assert len(run[f"{prefix}trials/params/x"].fetch_values()) == n_completed

        assert run[f"{prefix}trials/trials/0/user_attrs/dummy_trial_key"].fetch() == str(
            stringify_unsupported(dummy_user_attr)
        )
    else:
        assert not run.exists(f"{prefix}trials")

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completed_trials:
        assert run[f"{prefix}best/params"].fetch() == study.best_params
    else:
        assert not run.exists(f"{prefix}best/params")

    assert run[f"{prefix}study/user_attrs/dummy_study_key"].fetch() == str(stringify_unsupported(dummy_user_attr))

    assert run.exists(f"{prefix}study/study_name")
    assert run.exists(f"{prefix}study/distributions/")
