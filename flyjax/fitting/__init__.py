import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'evaluation',
        'evaluation_tools',
        'hierarchical',
        'joint',
        'train',
    },
    submod_attrs={
        'evaluation': [
            'log_likelihood_experiment',
            'negative_log_likelihood_experiment',
        ],
        'evaluation_tools': [
            'compute_aic',
            'compute_bic',
            'compute_waic',
            'cross_validation_score',
            'k_fold_split',
            'likelihood_ratio_test',
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
            'total_negative_log_likelihood_multi_group',
        ],
        'train': [
            'evaluate_model',
            'multi_start_train',
            'total_negative_log_likelihood',
            'train_model',
        ],
    },
)

__all__ = ['compute_aic', 'compute_bic', 'compute_waic',
           'cross_validation_score', 'evaluate_hierarchical_model',
           'evaluate_joint_model', 'evaluate_model', 'evaluation',
           'evaluation_tools', 'hierarchical', 'hierarchical_train_model',
           'joint', 'joint_train_model', 'k_fold_split',
           'likelihood_ratio_test', 'log_likelihood_experiment',
           'multi_start_hierarchical_train', 'multi_start_joint_train',
           'multi_start_train', 'negative_log_likelihood_experiment',
           'total_negative_log_likelihood',
           'total_negative_log_likelihood_multi_group',
           'total_nll_hierarchical', 'train', 'train_model']
