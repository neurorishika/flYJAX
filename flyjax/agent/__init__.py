import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'base',
        'rl',
    },
    submod_attrs={
        'base': [
            'herrnstein_agent',
            'learned_fixed_policy',
            'one_bit_agent',
            'random_policy',
            'test_agent',
        ],
        'rl': [
            'advanced_rl_agent',
            'dfq_agent_with_dual_lr',
            'dfq_agent_with_dynamic_exploration',
            'dfq_agent_with_epsilon_softmax',
            'dfq_agent_with_init',
            'differential_forgetting_q_agent',
            'differential_q_agent',
            'forgetting_q_agent',
            'q_agent',
            'zoo',
        ],
    },
)

__all__ = ['advanced_rl_agent', 'base', 'dfq_agent_with_dual_lr',
           'dfq_agent_with_dynamic_exploration',
           'dfq_agent_with_epsilon_softmax', 'dfq_agent_with_init',
           'differential_forgetting_q_agent', 'differential_q_agent',
           'forgetting_q_agent', 'herrnstein_agent', 'learned_fixed_policy',
           'one_bit_agent', 'q_agent', 'random_policy', 'rl', 'test_agent',
           'zoo']
