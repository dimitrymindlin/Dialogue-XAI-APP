# A.1 Feature Importances

**Question 1:** Among the following features, which one is the <i>most</i> important in influencing our machine learning
model’s
prediction (that is, variations in the value of that feature will most likely change the model’s prediction)?

1. {{a1_q1_1}}
2. {{a1_q1_2}}
3. {{a1_q1_3}}

**Question 2:** Among the following features, which one is the _least_ important in influencing our machine learning
model’s
prediction (that is, variations in the value of that feature will least likely change the model’s prediction)?

1. {{a1_q2_1}}
2. {{a1_q2_2}}
3. {{a1_q2_3}}

# A.2 Counterfactual Thinking

**Question 3:** Consider a person with the following profile:

| Attribute | Value |
|-----------|-------|

{% for key, value in a2_q1_1.items() -%}
| {{ key }} | {{ value }} |
{% endfor %}

Our machine learning model currently predicts this person will {{a2_q1_2}}. When all other features are kept the same,
which of the following changes on the attribute _{{a2_q1_3}}_ is most likely to change our model’s prediction (i.e.,
make the model
predict the person will **{{a2_q1_4}}**)?

1. {{a2_q1_5}}
2. {{a2_q1_6}}
3. {{a2_q1_7}}

**Question 4:** Consider a person with the following profile:

| Attribute | Value |
|-----------|-------|

{% for key, value in a2_q2_1.items() -%}
| {{ key }} | {{ value }} |
{% endfor %}

Our machine learning model currently predicts this person is {{a2_q2_2}}. If we change only one feature of this
profile but leave all other features unchanged, which of the following changes is going to change our model’s
prediction (i.e., make the model predict the person is {{a2_q2_3}})? Please check all that apply.

1. {{a2_q2_4}}
2. {{a2_q2_5}}
3. {{a2_q2_6}}
4. {{a2_q2_7}}
5. {{a2_q2_8}}

# A.3 Simulate Model Behavior

**Question 5:** Consider a person with the following profile:

| Attribute | Value |
|-----------|-------|

{% for key, value in a3_q1_1.items() -%}
| {{ key }} | {{ value }} |
{% endfor %}

What do you think our machine learning model will predict for this person?

1. The model will predict this person will {{a3_q1_2}}.
2. The model will predict this person will {{a3_q1_3}}.
3.

**Question 6:** Consider three people with the following profiles:

| Attribute | Person 1 | Person 2 | Person 3 |
|-----------|----------|----------|----------|

{% for key, value in a3_q2_1.items() -%}
| {{ key }} | {{ value }} | {{ a3_q2_2[key] }} | {{ a3_q2_3[key] }} |
{% endfor %}

For one of these three people, our machine learning model predicts that the person is {{a3_q2_4}}. Which one do
you think is this defendant?

1. Person 1
2. Person 2
3. Person 3