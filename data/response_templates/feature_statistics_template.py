def feature_statistics_template(feature_name,
                                mean,
                                std,
                                min_v,
                                max_v,
                                feature_units):
    ""
    # Check if feature has a unit
    unit = ""
    if feature_name in feature_units.keys():
        unit = feature_units[feature_name]

    if len(unit) > 0:
        mean = mean + " " + unit
        std = std + " " + unit
        min_v = min_v + " " + unit
        max_v = max_v + " " + unit

    return (f"Here are statistics for the feature <b>{feature_name}</b>: <br><br>"
            f"The <b>mean</b> is {mean}<br><br> one <b>standard deviation</b> is {std}<br><br>"
            f" the <b>minimum</b> value is {min_v}<br><br> and the <b>maximum</b> value is {max_v}.")
