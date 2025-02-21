import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'bootstrapping',
        'cv',
        'evaluation',
        'hierarchical',
        'joint',
        'model_comparison',
        'samplers',
        'train',
        'uncertainty',
    },
    submod_attrs={
        'bootstrapping': [
            'bootstrap_train_hierarchical',
            'bootstrap_train_joint',
            'bootstrap_train_single',
        ],
        'cv': [
            'k_fold_cross_validation_train',
            'k_fold_cross_validation_train_hierarchical',
            'k_fold_cross_validation_train_joint',
            'k_fold_split_experiments',
            'k_fold_split_subjects',
            'parallel_k_fold_cross_validation_train',
            'parallel_k_fold_cross_validation_train_hierarchical',
            'parallel_k_fold_cross_validation_train_joint',
            'run_cv_fold',
            'run_cv_fold_hierarchical',
            'run_cv_fold_joint',
        ],
        'evaluation': [
            'get_state_and_probs',
            'log_likelihood_experiment',
            'negative_log_likelihood_experiment',
            'total_negative_log_likelihood',
        ],
        'hierarchical': [
            'evaluate_hierarchical_model',
            'hierarchical_train_model',
            'multi_start_hierarchical_train',
            'total_nll_hierarchical',
        ],
        'joint': [
            'evaluate_joint_model',
            'joint_train_model',
            'multi_start_joint_train',
            'total_nll_multi_group',
        ],
        'model_comparison': [
            'aic',
            'bic',
            'likelihood_ratio_test',
        ],
        'samplers': [
            'base_randn_sampler',
            'base_uniform_sampler',
            'make_base_randn_sampler',
            'make_base_uniform_sampler',
        ],
        'train': [
            'evaluate_model',
            'multi_start_train',
            'train_model',
        ],
        'uncertainty': [
            'compute_hessian',
            'is_positive_definite',
            'laplace_uncertainty',
            'laplace_uncertainty_hierarchical',
            'laplace_uncertainty_joint',
        ],
    },
)

__all__ = ['aic', 'base_randn_sampler', 'base_uniform_sampler', 'bic',
           'bootstrap_train_hierarchical', 'bootstrap_train_joint',
           'bootstrap_train_single', 'bootstrapping', 'compute_hessian', 'cv',
           'evaluate_hierarchical_model', 'evaluate_joint_model',
           'evaluate_model', 'evaluation', 'get_state_and_probs',
           'hierarchical', 'hierarchical_train_model', 'is_positive_definite',
           'joint', 'joint_train_model', 'k_fold_cross_validation_train',
           'k_fold_cross_validation_train_hierarchical',
           'k_fold_cross_validation_train_joint', 'k_fold_split_experiments',
           'k_fold_split_subjects', 'laplace_uncertainty',
           'laplace_uncertainty_hierarchical', 'laplace_uncertainty_joint',
           'likelihood_ratio_test', 'log_likelihood_experiment',
           'make_base_randn_sampler', 'make_base_uniform_sampler',
           'model_comparison', 'multi_start_hierarchical_train',
           'multi_start_joint_train', 'multi_start_train',
           'negative_log_likelihood_experiment',
           'parallel_k_fold_cross_validation_train',
           'parallel_k_fold_cross_validation_train_hierarchical',
           'parallel_k_fold_cross_validation_train_joint', 'run_cv_fold',
           'run_cv_fold_hierarchical', 'run_cv_fold_joint', 'samplers',
           'total_negative_log_likelihood', 'total_nll_hierarchical',
           'total_nll_multi_group', 'train', 'train_model', 'uncertainty']
