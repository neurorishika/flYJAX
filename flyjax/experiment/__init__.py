import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'get',
    },
    submod_attrs={
        'get': [
            'fetch_behavioral_data',
            'get_experiments',
        ],
    },
)

__all__ = ['fetch_behavioral_data', 'get', 'get_experiments']
