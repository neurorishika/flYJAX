import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'get',
    },
    submod_attrs={
        'get': [
            'fetch_choices_and_rewards',
        ],
    },
)

__all__ = ['fetch_choices_and_rewards', 'get']
