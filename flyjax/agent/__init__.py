import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'model',
    },
    submod_attrs={
        'model': [
            'base_agent',
        ],
    },
)

__all__ = ['base_agent', 'model']
