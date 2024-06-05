import numpy as np
from isvd import iSVD

# U = rand(100,30)*rand(30,80); % low rank

U =np.random.rand(100,30) @ np.random.rand(30,80)

isvd = iSVD()
[Q,R,S] = isvd.Initialize(U[:,0])

for i in range(1, 80):
    isvd.Update(U[:,i]) # update the i-th column of U

[Q,R,S] = isvd.UpdateCheck() # finalize the iSVD


[Q_st,S_st,R_st] = np.linalg.svd(U,full_matrices=False)




# norm(Q*S*R' - U)
np.linalg.norm(Q @ S @ R.T - U)
# norm(Q_st*S_st*R_st' - U)
np.linalg.norm(Q_st @ np.diag(S_st) @ R_st - U)
# norm(abs(Q(:,end)' * W * Q(:,1)))
np.linalg.norm(np.abs(Q[:,-1].T @ isvd.W @ Q[:,0]))



