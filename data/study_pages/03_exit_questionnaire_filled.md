# A.1 Feature Importances

**Question 1:**
Among the following features, which one is the _most_ important in influencing our machine learning model’s prediction (that is, variations in the value of that feature will most likely change the model’s prediction)?

1. Checking account
2. Job Level
3. Age Group

**Question 2:** Among the following features, which one is the _least_ important in influencing our machine learning
model’s
prediction (that is, variations in the value of that feature will least likely change the model’s prediction)?

1. Gender
2. Credit amount
3. Credit Duration

# A.2 Counterfactual Thinking

**Question 3:** Consider a person with the following profile:

| Attribute | Value |
|-----------|-------|
| Age Group | young |
| Gender | male |
| Job Level | unskilled and resident |
| Housing Type | owns a house |
| Saving accounts | little |
| Checking account | little |
| Credit amount | 708 |
| Credit Duration | 12 |
| Credit Purpose | furniture/equipment |


Our machine learning model currently predicts this person will bad credit risk. When all other features are kept the same,
which of the following changes on the attribute _Saving accounts_ is most likely to change our model’s prediction (i.e.,
make the model
predict the person will **good credit risk**)?

1. Change Saving accounts to rich
2. Change Saving accounts to NA
3. Change Saving accounts to little

**Question 4:** Consider a person with the following profile:

| Attribute | Value |
|-----------|-------|
| Age Group | adult |
| Gender | male |
| Job Level | skilled |
| Housing Type | lives in housing that is free |
| Saving accounts | little |
| Checking account | little |
| Credit amount | 1333 |
| Credit Duration | 24 |
| Credit Purpose | education |


Our machine learning model currently predicts this person is bad credit risk. If we change only one feature of this
profile but leave all other features unchanged, which of the following changes is going to change our model’s
prediction (i.e., make the model predict the person is good credit risk)? Please check all that apply.

1. Change Saving accounts to quite rich
2. Change Checking account to rich
3. Change Credit amount to ADD_VALUE_HERE
4. Change Gender to female
5. Change Credit Duration to 12

# A.3 Simulate Model Behavior

**Question 5:** Consider a person with the following profile:

| Attribute | Value |
|-----------|-------|
| Age Group | student |
| Gender | female |
| Job Level | highly skilled |
| Housing Type | owns a house |
| Saving accounts | little |
| Checking account | moderate |
| Credit amount | 4795 |
| Credit Duration | 36 |
| Credit Purpose | radio/TV |


What do you think our machine learning model will predict for this person?

1. The model will predict this person will bad credit risk.
2. The model will predict this person will good credit risk.

**Question 6:** Consider three people with the following profiles:

| Attribute | Person 1 | Person 2 | Person 3 |
|-----------|----------|----------|----------|
| Age Group | adult | senior | senior |
| Gender | male | female | female |
| Job Level | highly skilled | skilled | skilled |
| Housing Type | owns a house | owns a house | lives in housing that is free |
| Saving accounts | little | little | little |
| Checking account | moderate | NA | little |
| Credit amount | 1209 | 1453 | 1333 |
| Credit Duration | 6 | 18 | 24 |
| Credit Purpose | business | business | car |


For one of these three people, our machine learning model predicts that the person is bad credit risk. Which one do
you think is this defendant?

1. Person 1
2. Person 2
3. Person 3