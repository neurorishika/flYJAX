import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'plotting',
    },
    submod_attrs={
        'plotting': [
            'plot_single_experiment_data',
            'plot_training_history',
        ],
    },
)

__all__ = ['plot_single_experiment_data', 'plot_training_history', 'plotting']
