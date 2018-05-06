import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set()

x = np.array([20,50,70,100,150,120,10,30,60,160,130,110])
y = np.array([15,44.7,64.5,94.2,144.3,114.1,5.3,25, 54.6, 154.1, 124.3, 104.4])
blah = x -y 
print(blah)
# plt.scatter(x, y)
plt.scatter(x,y)
plt.xlabel("true weight (g)")
plt.ylabel("scale reading (g)")
plt.title("calibrating triple beam balance via scale\n06 May 2018")
sns.regplot(x,y)
# sns.regplot(x,blah)
plt.show()
