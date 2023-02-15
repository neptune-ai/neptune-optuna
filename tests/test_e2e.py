import pytest

import optuna

try:
    # neptune-client=0.9.0+ package structure
    import neptune.new as neptune
except ImportError:
    # neptune-client>=1.0.0 package structure
    import neptune

import neptune_optuna.impl as npt_utils


@pytest.mark.parametrize("handler_namespace", [None, "handler_namespace"])
@pytest.mark.parametrize("base_namespace", [None, "base_namespace"])
def test_e2e(handler_namespace, base_namespace):

    n_trials = 5

    run = neptune.init_run()

    if handler_namespace is not None:
        handler = run[handler_namespace]
    else:
        handler = run

    neptune_callback = npt_utils.NeptuneCallback(handler, base_namespace=base_namespace)

    def objective(trial):
        x = trial.suggest_float("x", -10, 10)
        return (x - 2) ** 2

    study = optuna.create_study()
    study.optimize(objective, n_trials=n_trials, callbacks=[neptune_callback])

    validate_run(run, n_trials, study, handler_namespace, base_namespace)


def validate_run(run, n_trials, study, handler_namespace, base_namespace):

    prefix = ""
    if handler_namespace is not None:
        prefix = f"{handler_namespace}/"
    if base_namespace is not None:
        prefix = f"{prefix}{base_namespace}/"

    run.wait()

    assert run.exists(f"{prefix}best")
    assert run.exists(f"{prefix}study")
    assert run.exists(f"{prefix}trials")
    assert run.exists(f"{prefix}visualizations")

    run_structure = run.get_structure()
    if handler_namespace is not None:
        run_structure = run_structure[handler_namespace]
    if base_namespace is not None:
        run_structure = run_structure[base_namespace]
    assert len(run_structure["trials"]["trials"]) == n_trials

    assert len(run[f"{prefix}trials/values"].fetch_values()) == n_trials
    assert len(run[f"{prefix}trials/params/x"].fetch_values()) == n_trials

    assert run[f"{prefix}best/params"].fetch() == study.best_params

    assert run.exists("source_code/integrations/neptune-optuna")
