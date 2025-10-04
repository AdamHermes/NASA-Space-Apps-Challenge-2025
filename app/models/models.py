from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold, cross_val_predict
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, StackingClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)

# We'll evaluate each algorithm twice:
#  A) basic / initial hyperparams (paper's default-ish)
#  B) GridSearchCV tuning with a small grid (paper's grids were larger — adjust if you want more exhaustive search)

models_and_grids = {}

# AdaBoost
models_and_grids['AdaBoost'] = {
    'model': AdaBoostClassifier(random_state=42),
    'init_params': {'n_estimators':50, 'learning_rate':1.0},
    'grid': {'n_estimators':[50, 100, 500, 974], 'learning_rate':[1.0, 0.1]}
}

# Random Forest
models_and_grids['RandomForest'] = {
    'model': RandomForestClassifier(random_state=42, n_jobs=-1),
    'init_params': {'n_estimators':100, 'criterion':'gini', 'max_features':'sqrt'},
    'grid': {'n_estimators':[100, 400, 1600], 'criterion':['gini','entropy'], 'max_features':['sqrt','log2', None]}
}

# Extremely Randomized Trees (ExtraTrees)
models_and_grids['ExtraTrees'] = {
    'model': ExtraTreesClassifier(random_state=42, n_jobs=-1),
    'init_params': {'n_estimators':100, 'criterion':'gini', 'max_features':'sqrt'},
    'grid': {'n_estimators':[100,200,800], 'criterion':['gini','entropy'], 'max_features':['sqrt','log2', None]}
}

# Random Subspace Method -> use BaggingClassifier + DecisionTree with max_features subsetting (feature subspace)
# BaggingClassifier supports `max_features` for feature subsampling; base estimator decision tree
models_and_grids['RandomSubspace'] = {
    'model': BaggingClassifier(base_estimator=DecisionTreeClassifier(random_state=42),
                              random_state=42, n_jobs=-1),
    'init_params': {'n_estimators':10, 'max_samples':1.0, 'max_features':1.0},
    'grid': {'n_estimators':[10,100,1000], 'max_samples':[1.0, 0.1], 'max_features':[1.0, 0.1, 0.3]}
}

# Stacking - default initial: (RandomForest, GradientBoosting) + LogisticRegression as final estimator
base_estimators_initial = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
]
models_and_grids['Stacking'] = {
    'model': StackingClassifier(estimators=base_estimators_initial, final_estimator=LogisticRegression(max_iter=2000), n_jobs=-1, passthrough=False),
    'init_params': None,  # Stacking default is provided above
    # For tuning, we'll tune estimators' n_estimators indirectly by replacing with tuned rf/gb
    'grid': {
        # We'll allow alternative estimator sets (a small set of combinations)
        'estimators_choice': ['rf_gb', 'lgbm_gb']  # 'lgbm_gb' requires lightgbm; if not installed we'll fallback
    }
}

print("Ready to train: ", list(models_and_grids.keys()))
