def textual_cf(cfe_strings):
    output_string = ""

    for i, cfe_string in enumerate(cfe_strings):
        output_string += cfe_string
        output_string += f".<br><br>"

    return output_string
