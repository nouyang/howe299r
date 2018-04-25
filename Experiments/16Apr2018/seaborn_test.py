
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''
##
# Create a figure space matrix consisting of 3 columns and 2 rows
#
# Here is a useful template to use for working with subplots.
#
##################################################################
fig, ax = plt.subplots(figsize=(10,5), ncols=3, nrows=2)

left   =  0.125  # the left side of the subplots of the figure
right  =  0.9    # the right side of the subplots of the figure
bottom =  0.1    # the bottom of the subplots of the figure
top    =  0.9    # the top of the subplots of the figure
wspace =  .5     # the amount of width reserved for blank space between subplots
hspace =  1.1    # the amount of height reserved for white space between subplots

# This function actually adjusts the sub plots using the above paramters
plt.subplots_adjust(
        left    =  left, 
        bottom  =  bottom, 
        right   =  right, 
        top     =  top, 
        wspace  =  wspace, 
        hspace  =  hspace
)

# The amount of space above titles
y_title_margin = 1.2

plt.suptitle("Original vs Normalized vs Standardized", y = 1.09, fontsize=20)

### Bathrooms
ax[0][0].set_title("Original",                    y = y_title_margin)
ax[0][1].set_title("Normalized (MinMax)",         y = y_title_margin)
ax[0][2].set_title("Starndardized (StardardScaler)", y = y_title_margin)

sns.distplot(df['bathrooms'],        kde = False, ax=ax[0][0])
sns.distplot(df['norm_bathrooms'],   kde = False, ax=ax[0][1])
sns.distplot(df['stand_bathrooms'],  kde = False, ax=ax[0][2])

# Set all labels on the row axis of subplots for bathroom data to "bathrooms"
[ax[0][i].set_xlabel("bathrooms") for i in range(0, 3)]

### Square feet
ax[1][0].set_title("Original",                  y = y_title_margin)
ax[1][1].set_title("Normalized (MinMax)",       y = y_title_margin)
ax[1][2].set_title("Standardized (StandardScaler)", y=y_title_margin)

sns.distplot(df['square_feet'],      kde = False, ax=ax[1][0])
sns.distplot(df['norm_square_feet'], kde = False, ax=ax[1][1])
sns.distplot(df['stand_square_feet'],kde = False, ax=ax[1][2])

# Set all labels on the row axis of subplots for square_feet data to "square_feet"
[ax[1][i].set_xlabel("square_feet") for i in range(0, 3)]
'''

'''
sns.set(color_codes=True)
tips = sns.load_dataset("tips")
g = sns.lmplot(x="total_bill", y="tip", data=tips)
g = sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips)
g = sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips, 
               palette=dict(Yes="g", No="m"))
plt.show()
'''


