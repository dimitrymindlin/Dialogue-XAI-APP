# The model predicts {{model_prediction}} for the current {{instance_type}}.
## Here are explanations for the prediction:

### The most important features for the decision are the following, ordered by importance:


| Attribute | Importance Score |
|-----------|------------------|
{% for key, value in feature_importance.items() -%}
| {{ key }} | {{ "%.4f"|format(value) }} |
{% endfor %}

### The model would predict {{opposite_class}} if the following features were changed:

{{counterfactuals}}

### As long as the following features are kept the same, the model will predict {{model_prediction}}:

{{anchors}}.