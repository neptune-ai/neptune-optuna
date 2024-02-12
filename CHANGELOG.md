## neptune-optuna 1.2.0

### Features
- Added `log_all_trials` flag in `NeptuneCallback` ([#41](https://github.com/neptune-ai/neptune-optuna/pull/41))

## neptune-optuna 1.1.0

### Changes
- Removed `neptune` and `neptune-client` from base requirements
- Updated for compatibility with `neptune 1.X`

## neptune-optuna 1.0.0

### Fixes

- Fixing bug in `log_study_metadata` introduced in the previous release.

### Breaking changes

- We are not tracing anymore `study/system_attrs` that was deprecated by Optuna.

## neptune-optuna 0.11.0

We updated the integration to use the most recent version of the Neptune client.


## neptune-optuna 0.10.2

### Fixes
- Fixed plot_intermeadiate_values() got unexpected keyword argument error ([#28](https://github.com/neptune-ai/neptune-optuna/pull/28))

## neptune-optuna 0.10.1

### Fixes
- Fixed scikit-learn requirement package name ([#26](https://github.com/neptune-ai/neptune-optuna/pull/26))

## neptune-optuna 0.10.0

### Changes
- Move `neptune_optuna` package to `src` directory ([#18](https://github.com/neptune-ai/neptune-optuna/pull/18))
- Poetry as a package builder ([#24](https://github.com/neptune-ai/neptune-optuna/pull/24))

### Features
- Add support for multi-objective training ([#14](https://github.com/neptune-ai/neptune-optuna/pull/14))

### Fixes
- Fixed NeptuneCallback import error - now possible to directly import with `from neptune_optuna import NeptuneCallback` ([#21](https://github.com/neptune-ai/neptune-optuna/pull/21))

## neptune-optuna 0.9.15

### Changes
- Changed integrations utils to be imported from non-internal package ([#17](https://github.com/neptune-ai/neptune-optuna/pull/17))

## neptune-optuna 0.9.14

### Features

- Mechanism to prevent using legacy Experiments in new-API integrations ([#6](https://github.com/neptune-ai/neptune-optuna/pull/6))

### Fixes

- Added more checks to prevent exceptions ([#11](https://github.com/neptune-ai/neptune-optuna/pull/11))
- Disabled plotting parameters importance on failure ([#10](https://github.com/neptune-ai/neptune-optuna/pull/10))

## neptune-optuna 0.9.13

### Fixes

- Handling exceptions and trial pruning ([#5](https://github.com/neptune-ai/neptune-optuna/pull/5))
