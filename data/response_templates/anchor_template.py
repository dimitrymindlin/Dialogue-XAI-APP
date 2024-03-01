def anchor_template(exp, template_manager):
    output_string = ""
    # output_string += "By fixing all of the following attributes, the prediction stays the same even though other attributes are changed:"
    output_string += "<br><br>"

    display_names_exp_names = []
    for idx, change_string in enumerate(exp.names()):
        feature = change_string.split(" ")[0]
        value = change_string.split(" ")[2]
        # If its a categorical feature, get the display name of the value
        """if feature in template_manager.encoded_col_mapping.keys(): NOT NEEDED?!
            value = template_manager.get_encoded_feature_name(feature, value)"""
        display_name = template_manager.get_feature_display_name_by_name(feature)
        display_names_exp_names.append(f"{display_name} {change_string.split(' ')[1]} {value}")

    explanation_text = " and ".join(display_names_exp_names).replace("<=", "is not above")
    explanation_text = explanation_text.replace(">=", "is not below")
    explanation_text = explanation_text.replace(">", "is above")
    explanation_text = explanation_text.replace("<", "is below")

    explanation_text = explanation_text.replace(".00", "")  # remove decimal zeroes
    explanation_text += "</b>"
    explanation_text = "<b>" + explanation_text
    return explanation_text
