# Neptune + Optuna integration

> [!IMPORTANT]
> This integration works with Neptune app version `2.x`, which uses Python client version `1.x`.<br/>

Neptune is an experiment tracker purpose-built for foundation model training.

With Neptune, you can monitor thousands of per-layer metrics—losses, gradients, and activations—at any scale. Visualize them with no lag and no missed spikes. Drill down into logs and debug training issues fast. Keep your model training stable while reducing wasted GPU cycles.

This integration lets you use it as an Optuna visualization dashboard to log and monitor hyperparameter sweeps live.

## What will you get with this integration?

* Log and monitor the Optuna hyperparameter sweep live:
  * values and params for each Trial
  * best values and params for the Study
  * hardware consumption and console logs
  * interactive plots from the optuna.visualization module
  * parameter distributions for each Trial
  * Study object itself for 'InMemoryStorage' or the database location for the Studies with database storage
* Load the Study directly from the existing Neptune run

![image](https://docs-legacy.neptune.ai/img/app/integrations/optuna.png)

## Resources

* [Documentation](https://docs-legacy.neptune.ai/integrations/optuna)
* [Code example on GitHub](https://github.com/neptune-ai/examples/blob/main/integrations-and-supported-tools/optuna/scripts)
* [Run logged in the Neptune app](https://app.neptune.ai/o/common/org/optuna-integration/runs/details?viewId=b6190a29-91be-4e64-880a-8f6085a6bb78&detailsTab=dashboard&dashboardId=Vizualizations-5ea92658-6a56-4656-b225-e81c6fbfc8ab&shortId=NEP1-18517&type=run)
* [Run example in Google Colab](https://colab.research.google.com/github/neptune-ai/examples/blob/master/integrations-and-supported-tools/optuna/notebooks/Neptune_Optuna_integration.ipynb)

## Example

On the command line:

```
pip install neptune-optuna
```

In Python:

```python
import neptune
import neptune.integrations.optuna as npt_utils

# Start a run
run = neptune.init_run(
    api_token=neptune.ANONYMOUS_API_TOKEN,
    project="common/optuna-integration",
)

# Create a NeptuneCallback instance
neptune_callback = npt_utils.NeptuneCallback(run)

# Pass the callback to study.optimize()
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, callbacks=[neptune_callback])

# Watch the optimization live in Neptune
```

## Support

If you got stuck or simply want to talk to us, here are your options:

* Check our [FAQ page](https://docs-legacy.neptune.ai/getting_help)
* You can submit bug reports, feature requests, or contributions directly to the repository.
* Chat! When in the Neptune application click on the blue message icon in the bottom-right corner and send a message. A real person will talk to you ASAP (typically very ASAP),
* You can just shoot us an email at support@neptune.ai
