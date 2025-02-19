import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'agent',
        'experiment',
        'fitting',
        'simulation',
        'utils',
    },
    submod_attrs={
        'agent': [
            'advanced_rl_agent',
            'base',
            'dfq_agent_with_dual_lr',
            'dfq_agent_with_epsilon_softmax',
            'dfq_agent_with_init',
            'differential_forgetting_q_agent',
            'differential_q_agent',
            'forgetting_q_agent',
            'learned_fixed_policy',
            'q_agent',
            'random_policy',
            'rl',
            'test_agent',
            'zoo',
        ],
        'experiment': [
            'fetch_behavioral_data',
            'get',
            'get_experiments',
        ],
        'fitting': [
            'aic',
            'base_randn_sampler',
            'base_uniform_sampler',
            'bic',
            'cv',
            'evaluate_hierarchical_model',
            'evaluate_joint_model',
            'evaluate_model',
            'evaluation',
            'hierarchical',
            'hierarchical_train_model',
            'joint',
            'joint_train_model',
            'k_fold_cross_validation_train',
            'k_fold_cross_validation_train_hierarchical',
            'k_fold_cross_validation_train_joint',
            'k_fold_split_experiments',
            'k_fold_split_subjects',
            'likelihood_ratio_test',
            'log_likelihood_experiment',
            'make_base_randn_sampler',
            'make_base_uniform_sampler',
            'model_comparison',
            'multi_start_hierarchical_train',
            'multi_start_joint_train',
            'multi_start_train',
            'negative_log_likelihood_experiment',
            'parallel_k_fold_cross_validation_train',
            'parallel_k_fold_cross_validation_train_hierarchical',
            'parallel_k_fold_cross_validation_train_joint',
            'run_cv_fold',
            'run_cv_fold_hierarchical',
            'run_cv_fold_joint',
            'samplers',
            'total_negative_log_likelihood',
            'total_nll_multi_group',
            'total_nll_hierarchical',
            'train',
            'train_model',
        ],
        'simulation': [
            'parse',
            'parse_reward_matrix',
            'simulate',
            'simulate_dataset',
            'simulate_dataset_different_params',
            'simulate_dataset_jit',
            'simulate_dataset_jit_different_params',
            'simulate_experiment',
            'simulate_experiment_jit',
        ],
        'utils': [
            'plot_single_experiment_data',
            'plot_training_history',
            'plotting',
        ],
    },
)

__all__ = ['advanced_rl_agent', 'agent', 'aic', 'base', 'base_randn_sampler',
           'base_uniform_sampler', 'bic', 'cv', 'dfq_agent_with_dual_lr',
           'dfq_agent_with_epsilon_softmax', 'dfq_agent_with_init',
           'differential_forgetting_q_agent', 'differential_q_agent',
           'evaluate_hierarchical_model', 'evaluate_joint_model',
           'evaluate_model', 'evaluation', 'experiment',
           'fetch_behavioral_data', 'fitting', 'forgetting_q_agent', 'get',
           'get_experiments', 'hierarchical', 'hierarchical_train_model',
           'joint', 'joint_train_model', 'k_fold_cross_validation_train',
           'k_fold_cross_validation_train_hierarchical',
           'k_fold_cross_validation_train_joint', 'k_fold_split_experiments',
           'k_fold_split_subjects', 'learned_fixed_policy',
           'likelihood_ratio_test', 'log_likelihood_experiment',
           'make_base_randn_sampler', 'make_base_uniform_sampler',
           'model_comparison', 'multi_start_hierarchical_train',
           'multi_start_joint_train', 'multi_start_train',
           'negative_log_likelihood_experiment',
           'parallel_k_fold_cross_validation_train',
           'parallel_k_fold_cross_validation_train_hierarchical',
           'parallel_k_fold_cross_validation_train_joint', 'parse',
           'parse_reward_matrix', 'plot_single_experiment_data',
           'plot_training_history', 'plotting', 'q_agent', 'random_policy',
           'rl', 'run_cv_fold', 'run_cv_fold_hierarchical',
           'run_cv_fold_joint', 'samplers', 'simulate', 'simulate_dataset',
           'simulate_dataset_different_params', 'simulate_dataset_jit',
           'simulate_dataset_jit_different_params', 'simulate_experiment',
           'simulate_experiment_jit', 'simulation', 'test_agent',
           'total_negative_log_likelihood',
           'total_nll_multi_group',
           'total_nll_hierarchical', 'train', 'train_model', 'utils', 'zoo']
