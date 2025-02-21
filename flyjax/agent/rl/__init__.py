import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'zoo',
    },
    submod_attrs={
        'zoo': [
            'advanced_rl_agent',
            'dfq_agent_with_dual_lr',
            'dfq_agent_with_dynamic_exploration',
            'dfq_agent_with_epsilon_softmax',
            'dfq_agent_with_init',
            'differential_forgetting_q_agent',
            'differential_q_agent',
            'forgetting_q_agent',
            'q_agent',
        ],
    },
)

__all__ = ['advanced_rl_agent', 'dfq_agent_with_dual_lr',
           'dfq_agent_with_dynamic_exploration',
           'dfq_agent_with_epsilon_softmax', 'dfq_agent_with_init',
           'differential_forgetting_q_agent', 'differential_q_agent',
           'forgetting_q_agent', 'q_agent', 'zoo']
