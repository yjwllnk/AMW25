import seaborn as sns
import shap

def plot_corr(df):
    sns.clustermap(df_corr.corr(), cmap='coolwarm', annot=False, vmin=-1, vmax=1, figsize=(10, 10))
    pass

def plot_shap(**kwargs):
    X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=1/6,random_state=random_seed)

    shap_values = shap.TreeExplainer(model).shap_values(X_test)
    shap_interaction_values = shap.TreeExplainer(model).shap_interaction_values(X_test)
    shap.summary_plot(shap_values[:,:,2], X_test,plot_type='violin',feature_names=X.columns,max_display=50)
    pass
