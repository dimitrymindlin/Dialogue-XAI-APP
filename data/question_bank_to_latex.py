# Read the CSV file
import pandas as pd

df = pd.read_csv('question_bank.csv', delimiter=';')

# Delete the first two columns
df = df.drop(df.columns[[-1, -2]], axis=1)

# Adjust the column width
column_width = 40  # Modify this value as needed

# Format the column specifications for LaTeX table
# column_spec = '|'.join(['p{' + str(column_width) + 'mm}' for _ in df.columns])


# Print the last two columns in LaTeX table format
latex_table = df.to_latex(index=False, escape=False, header=True)

# Print the LaTeX table
# Add font size command
latex_table = latex_table.replace(r'\begin{tabular}', r'\begin{tabular}{\small}')
latex_table = latex_table.replace(r'\end{tabular}', r'\end{tabular}')

print(latex_table)
# save the table to a file
with open('question_bank.tex', 'w') as f:
    f.write(latex_table)
