# !/user/bin/env python
# -*- coding : utf-8 -*-

import numpy as np

a = np.array([[1,2,3],[1,8,10],[1,2,3]])


pos = np.where(a >= 0.8* np.max(a))
pos = np.array(pos)
pos = np.transpose(pos)
print(pos)