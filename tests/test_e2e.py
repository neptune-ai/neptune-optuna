import optuna

try:
    # neptune-client=0.9.0+ package structure
    import neptune.new as neptune
except ImportError:
    # neptune-client>=1.0.0 package structure
    import neptune

import neptune_optuna.impl as npt_utils

def test_e2e():

    n_trials = 5
    # base_namespace = "optuna"

    run = neptune.init_run()
    # handler = run[base_namespace]
    neptune_callback = npt_utils.NeptuneCallback(run)

    def objective(trial):
        x = trial.suggest_float('x', -10, 10)
        return (x - 2) ** 2

    study = optuna.create_study()
    study.optimize(objective, n_trials=n_trials, callbacks=[neptune_callback])

    run.wait()

    assert run.exists("best")
    assert run.exists("study")
    assert run.exists("trials")
    assert run.exists("visualizations")

    run_structure = run.get_structure()

    assert len(run_structure["trials"]["trials"]) == n_trials
    assert len(run["trials/values"].fetch_values()) == n_trials
    assert len(run["trials/params/x"].fetch_values()) == n_trials

    assert run["best/params"].fetch() == study.best_params

    assert run.exists("source_code/integrations/neptune-optuna")
