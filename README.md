# Neptune + Optuna integration

Neptune is a tool for experiment tracking, model registry, data versioning, and monitoring model training live.

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

![image](https://user-images.githubusercontent.com/97611089/160636423-82951249-a5d8-40d3-be34-4c2ff470b9db.png)
*Parallel coordinate plot logged to Neptune*

## Resources

* [Documentation](https://docs.neptune.ai/integrations/optuna)
* [Code example on GitHub](https://github.com/neptune-ai/examples/blob/main/integrations-and-supported-tools/optuna/scripts)
* [Runs logged in the Neptune app](https://app.neptune.ai/o/common/org/optuna-integration/experiments?split=bth&dash=parallel-coordinates-plot&viewId=b6190a29-91be-4e64-880a-8f6085a6bb78)
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

* Check our [FAQ page](https://docs.neptune.ai/getting_help)
* You can submit bug reports, feature requests, or contributions directly to the repository.
* Chat! When in the Neptune application click on the blue message icon in the bottom-right corner and send a message. A real person will talk to you ASAP (typically very ASAP),
* You can just shoot us an email at support@neptune.ai
