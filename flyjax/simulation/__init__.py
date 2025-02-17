import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'parse',
        'simulate',
    },
    submod_attrs={
        'parse': [
            'parse_reward_matrix',
        ],
        'simulate': [
            'simulate_dataset',
            'simulate_dataset_different_params',
            'simulate_dataset_jit',
            'simulate_dataset_jit_different_params',
            'simulate_experiment',
            'simulate_experiment_jit',
        ],
    },
)

__all__ = ['parse', 'parse_reward_matrix', 'simulate', 'simulate_dataset',
           'simulate_dataset_different_params', 'simulate_dataset_jit',
           'simulate_dataset_jit_different_params', 'simulate_experiment',
           'simulate_experiment_jit']
