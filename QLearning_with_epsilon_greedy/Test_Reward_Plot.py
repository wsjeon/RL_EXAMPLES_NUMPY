import numpy as np
import matplotlib.pyplot as plt

Test_Reward=np.load('DATA__GM_0.99_LR_0.1_K_1_SD_0_TE_0.npy')

print Test_Reward.shape

for i in range(0,Test_Reward.shape[2],1):
    A=np.mean(Test_Reward[:, :, i], axis=0)
    print A
    plt.plot(np.arange(0,5001,100),np.concatenate([[0],A]),'-')

# print Test_Reward[:,:,0]

plt.show()
