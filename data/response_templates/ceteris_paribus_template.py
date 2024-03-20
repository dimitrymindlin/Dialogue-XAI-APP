def cp_categorical_template(feature_name, opposite_class, categories):
    answer_text = f"The following changes to {feature_name} will switch the prediction to <b>{opposite_class}</b>: <br>"
    for i, category in enumerate(categories):
        answer_text += f"<b>{category}</b>"
        if i + 1 != len(categories):
            answer_text += ", "
        else:
            answer_text += "."
    return answer_text


def cp_numerical_template(feature_name, opposite_class, sign, x_flip_value, template_manager):
    feature_display_name = template_manager.get_feature_display_name_by_name(feature_name)
    if x_flip_value is not None:
        rounded_value = round(x_flip_value, 2)
        return f"Altering {feature_display_name} can change the prediction of the model: {sign} <b>{feature_display_name}</b> to" \
               f" <b>{rounded_value}</b> will change the prediction to <b>{opposite_class}</b>."
    else:
        return f"No changes in <b>{feature_display_name}</b> alone can change the model prediction."