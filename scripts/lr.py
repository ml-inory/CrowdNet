# -*- coding: utf-8 -*-
''' linear regression '''
import sys
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

if len(sys.argv) < 3:
    print 'Usage: python lr.py x.txt y.txt'
    sys.exit(-1)
    
x = np.loadtxt(open(sys.argv[1],'r')).reshape(-1,1)
y = np.loadtxt(open(sys.argv[2],'r'))

# 建立线性回归模型
regr = linear_model.LinearRegression()

# 拟合
regr.fit(x, y)

# 不难得到直线的斜率、截距
k, b = regr.coef_[0], regr.intercept_

print 'k = {}, b = {}'.format(k, b)

plt.plot(x, y, 'bx')
plt.plot(x, regr.predict(x), color='red', label='k={} b={}'.format(k, b))
plt.legend(loc='best')

plt.savefig('predict_line.png')
plt.show()