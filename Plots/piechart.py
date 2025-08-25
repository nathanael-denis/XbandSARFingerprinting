import matplotlib.pyplot as plt

# Data for the pie chart
total_data_tb = 3.76  # Total dataset size in terabytes
usrp1_data_tb = 2.48  # Data collected by USRP 1 in terabytes
usrp2_data_tb = 1.28  # Data collected by USRP 2 in terabytes
labels = ['USRP 1', 'USRP 2']
sizes = [usrp1_data_tb, usrp2_data_tb]
percentages = [100 * usrp1_data_tb / total_data_tb, 100 * usrp2_data_tb / total_data_tb]
colors = ["#589fc9", "#52c474"]  # Blue and orange for contrast
explode = (0.05, 0)  # Slightly explode USRP 1 slice for emphasis

# Create pie chart with doubled font sizes, no title or text
plt.figure(figsize=(8, 6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.2f%%', 
        shadow=False, startangle=90, textprops={'fontsize': 32})
plt.axis('equal')  # Equal aspect ratio ensures pie chart is circular

# Save the figure as PNG
plt.savefig('data_repartition_no_title_text.png', format='png', dpi=300, bbox_inches='tight')
plt.close()