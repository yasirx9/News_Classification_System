import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('trainingDataset.csv')

# Count number of news articles per category before dropping NaN
category_count_before = data['Category'].value_counts()

# Handle missing values (drop NaN in 'News' column)
data_clean = data.dropna(subset=['News'])

# Count number of news articles per category after dropping NaN
category_count_after = data_clean['Category'].value_counts()

# Create a DataFrame for plotting
df = pd.DataFrame({
    'Category': category_count_before.index,
    'Before DropNA': category_count_before.values,
    'After DropNA': category_count_after.reindex(category_count_before.index, fill_value=0).values
})

# Plotting using pure matplotlib
fig, ax = plt.subplots(figsize=(10, 6))

# Create a bar plot with before and after dropna
bars1 = ax.bar(df['Category'], df['Before DropNA'], label='Before DropNA', color='lightblue')
bars2 = ax.bar(df['Category'], df['After DropNA'], label='After DropNA', color='lightgreen', bottom=df['Before DropNA'])

# Annotate the bars with the number of news articles
for bar in bars1:
    ax.annotate(
        f'{int(bar.get_height())}',  # The number to annotate
        (bar.get_x() + bar.get_width() / 2, bar.get_height()),  # Position of the text
        ha='center', va='center',  # Horizontal and vertical alignment
        fontsize=12, color='black',  # Font size and color
        xytext=(0, 5),  # Offset for better visibility
        textcoords='offset points'  # Use offset points to move the text slightly above
    )

for bar in bars2:
    ax.annotate(
        f'{int(bar.get_height())}',  # The number to annotate
        (bar.get_x() + bar.get_width() / 2, bar.get_height() + bar.get_y()),  # Position of the text
        ha='center', va='center',  # Horizontal and vertical alignment
        fontsize=12, color='black',  # Font size and color
        xytext=(0, 5),  # Offset for better visibility
        textcoords='offset points'  # Use offset points to move the text slightly above
    )

# Title and labels
ax.set_title('Category vs Number of News Articles (Before and After DropNa)', fontsize=16)
ax.set_xlabel('Category', fontsize=14)
ax.set_ylabel('Number of News Articles', fontsize=14)

# Rotate x-axis labels for readability
plt.xticks(rotation=45)
plt.tight_layout()

# Add a legend
ax.legend()

# Show the plot
plt.show()
