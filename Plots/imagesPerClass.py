import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

# Original data
satellites = ['X2', 'X4', 'X5', 'X6', 'X7', 'X8', 'X11', 'X13', 'X14', 'X15', 
              'X17', 'X20', 'X21', 'X23', 'X25', 'X26', 'X27', 'X30', 'X31', 'X33', 
              'X34', 'X35', 'X36', 'X38', 'X39', 'X40', 'X41', 'X42', 'X43', 'X44', 
              'X45', 'X46', 'X47', 'X48', 'X49', 'X50', 'X51']
file_counts = [1773, 609, 1600, 2275, 1705, 1240, 1276, 1357, 2157, 975, 
               991, 1521, 1513, 2038, 1149, 1083, 1912, 1365, 1623, 1104, 
               1152, 1505, 1490, 1632, 1882, 913, 2518, 1807, 2209, 2121, 
               1331, 1171, 1593, 1850, 673, 896, 1551]

# Sort by file count descending
sorted_data = sorted(zip(satellites, file_counts), key=lambda x: x[1], reverse=True)

grouped_satellites = []
grouped_counts = []

group = [sorted_data[0][0]]
group_counts = [sorted_data[0][1]]

for i in range(1, len(sorted_data)):
    current_sat, current_count = sorted_data[i]
    last_count = group_counts[-1]
    if abs(current_count - last_count) <= 50:
        # Close enough to group
        group.append(current_sat)
        group_counts.append(current_count)
    else:
        # Save previous group
        grouped_satellites.append("(" + ",".join(group) + ")")
        grouped_counts.append(sum(group_counts) / len(group_counts))  # average count for group
        # Start new group
        group = [current_sat]
        group_counts = [current_count]

# Add last group
grouped_satellites.append("(" + ",".join(group) + ")")
grouped_counts.append(sum(group_counts) / len(group_counts))

plt.figure(figsize=(10, max(4, len(grouped_satellites)*0.3)))  # smaller height based on number of bars
y_positions = range(len(grouped_satellites))
bars = plt.barh(y_positions, grouped_counts, height=0.3, color="#62d989", align='center')  # thinner bars

plt.figure(figsize=(10, max(4, len(grouped_satellites)*0.3)))  # smaller height based on number of bars
y_positions = range(len(grouped_satellites))
bars = plt.barh(y_positions, grouped_counts, height=0.3, color="#62d989", align='center')  # thinner bars

# Add cross markers at the end of each bar
for y, x in zip(y_positions, grouped_counts):
    plt.plot(x, y, marker='x', markersize=12, color='black')

# Add satellite group labels to the left of each bar (font size unchanged)
max_count = max(grouped_counts)
for y, x, label in zip(y_positions, grouped_counts, grouped_satellites):
    plt.text(x + max_count * 0.01, y, label, va='center', ha='left', fontsize=12, fontweight='bold',
             path_effects=[path_effects.withStroke(linewidth=3, foreground='white')])

plt.yticks(ticks=y_positions, labels=[""] * len(grouped_satellites))
plt.xticks(fontsize=14, fontweight='bold',
           path_effects=[path_effects.withStroke(linewidth=3, foreground='white')])

plt.xlim(left=500, right=max_count * 1.07)  # push right border a bit
plt.ylim(-0.5, len(grouped_satellites) - 0.5)
plt.tight_layout()
plt.savefig('satellite_files_horizontal_bars_grouped_compact_offset_right.png', format='png', dpi=300, bbox_inches='tight')
plt.close()