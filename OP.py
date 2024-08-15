import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

class MonteCarlo:
    def __init__(self):
        self.Spath = None
    def simulations(self, S0, r, T, sigma, I,  steps, dn = 0, plotpath = False, plothist = False, plotreturnhist = False):
        """

        :param S0: 初始价格
        :param r: 无风险收益率
        :param T: 到期期限（年）
        :param sigma: 波动率
        :param I: 路径
        :param dn: 敲入点
        :param steps:
        :param plotpath:
        :param plothist:
        :return:

        """
        delta_t = float(T)/steps
        Spath = np.zeros((int(steps + 1), int(I)))
        Spath[0] = S0

        for t in range(1, steps + 1):
            z = np.random.standard_normal(I)
            middle1 = Spath[t-1, 0:I] * np.exp((r - 0.5 * sigma ** 2) * delta_t + sigma * np.sqrt(delta_t) * z)
            # 去掉超过涨跌停的值
            uplimit = Spath[t-1] * 1.1
            lowlimit = Spath[t-1] * 0.9
            temp = np.where(uplimit < middle1, uplimit, middle1)
            temp = np.where(lowlimit > middle1, lowlimit, temp)
            Spath[t, 0:I] = temp
        if plotpath:
            plt.plot(Spath[:, :])
            plt.plot([dn]*len(Spath))
            plt.xlabel('time')
            plt.ylabel('price')
            plt.title('Price Simulation')
            plt.grid(True)
            plt.show()
            plt.close()

        if plothist:
            plt.hist(Spath[int(T*steps)], bins=50)
            plt.title('T moment price Histogram')
            plt.show()
            plt.close()

        if plotreturnhist:
            returns = np.diff(np.log(Spath), axis=0)
            plt.hist(returns[-1], bins=50)
            plt.title('T Moment Return Histogram')
            plt.grid(True)
            plt.show()
            plt.close()

        self.Spath = Spath

        

    def accumulator(self, S, KO, r, n, m, I, type = 'continue'):
        '''
        S: Strike price 行使价
        KO: Knock out price 取消价
        n: quantity to buy per time
        I: simulation times 模拟次数
        type: whether continue when knock out
        m: mutiplier when price fall below strike price
        '''
        Spath = self.Spath #加载模拟好的路径
        [row, column] = Spath.shape
        payoff = np.zeros(I)
        
        for i in range(column):
            payoff[i] = 0
            for j in range(row):
                price = Spath[j,i]
                if price >= KO:
                    payoff[i] += 0     
                    if type == 'end':
                        break
                elif price < KO and price > S:
                    payoff[i] += (price - S)*np.exp(-r*(j+1)/252)*n #注意这里的r是年化的无风险利率，日折现需要除以
                else:
                    payoff[i] += (price - S)*np.exp(-r*(j+1)/252)*n*m


        aq_price = np.mean(payoff)
        return aq_price
    
    def decumulator(self, S, KO, r, n, m, I, type = 'continue'):
        '''
        S0: 初始价格
        S: Strike price 行使价
        KO: Knock out price 取消价

        '''
        # TO MODIFY
        Spath = self.Spath
        [row, column] = Spath.shape
        payoff = np.zeros(I)
        
        for i in range(column):
            payoff[i] = 0
            for j in range(row):
                price = Spath[j,i]
                if price <= KO:
                    payoff[i] += 0     
                    if type == 'end':
                        break
                elif price > KO and price < S:
                    payoff[i] += (S - price)*np.exp(r*(j+1))*n
                else:
                    payoff[i] += (S - price)*np.exp(r*(j+1))*n*m    

        dq_price = np.mean(payoff)
        return dq_price
    
    def snowball_cashflow(self, coupon, I, k_out, lock_period, dn):
        price_path = self.Spath
        payoff = np.zeros(I)
        knock_out_times = 0
        knock_in_times = 0
        existence_times = 0
        for i in range(I):
            # 收盘价超过敲出线的交易日
            tmp_up_d = np.where(price_path[:, i] > k_out)
            # 收盘价超出敲出线的观察日
            tmp_up_m = tmp_up_d[0][tmp_up_d[0] % 21 == 0]
            # 收盘价超出敲出线的观察日（超过封闭期）
            tmp_up_m_md = tmp_up_m[tmp_up_m > lock_period]
            tmp_dn_d = np.where(price_path[:, i] < dn)
            # 根据合约条款判断现金流

            # 情景1：发生过向上敲出
            if len(tmp_up_m_md) > 0:
                t = tmp_up_m_md[0]
                payoff[i] = coupon * (t/252) * np.exp(-r * t/252)
                knock_out_times += 1

            # 情景2：未敲出且未敲入
            elif len(tmp_up_m_md) == 0 and len(tmp_dn_d[0]) == 0:
                payoff[i] = coupon * np.exp(-r * T)
                existence_times += 1

            # 情景3：只发生向下敲入，不发生向上敲出
            elif len(tmp_dn_d[0]) > 0 and len(tmp_up_m_md) == 0:
                # 只有向下敲入，没有向上敲出
                payoff[i] = 0 if price_path[len(price_path)-1][i] > 1 else (price_path[len(price_path)-1][i] - S) * np.exp(-r * T)
                knock_in_times += 1
            else:
                print(i)
        return payoff, knock_out_times, knock_in_times, existence_times
    
    def snowball(self, coupon, I, k_out, lock_period, dn):
        price_path = self.Spath
        payoff, *_ = self.snowball_cashflow(coupon, I, k_out, lock_period, dn)
        price = sum(payoff)/len(payoff)
        return price
    
    def call(self, S):
        '''
        S: Strike Price
        '''
        Spath = self.Spath
        St = Spath[-1]
        call_price = np.maximum(St-S, 0)
        return call_price

    def put(self, S):
        '''
        S: Strike Price
        '''
        Spath = self.Spath
        St = Spath[-1]
        put_price = np.maximum(S-St, 0)

    def seagull(self, K1, K2, K3, type = 'call'):
        '''
        K1: Strike price of long call/put
        K2: Strike price of short call/put
        K3: Strike price of short put/call
        '''
        if type == 'call':
            seagull_price = self.call(K1)-self.call(K2)-self.put(K3)
        elif type == 'put':
            seagull_price = self.put(K1)-self.put(K2)-self.call(K3)
        else:
            raise ValueError('enter the correct option type')
        
        return seagull_price
        
        
        

if __name__ == '__main__':
    
    S0 = 5658
    S = 5742
    # S = S0+3343
    KO_1 = 5558
    KO_2 = 5558
    # KO_2 = S0-3343
    r = 0.05
    T = 30/252
    sigma = 0.115
    steps = 30
    n = 1/steps
    m = 3
    I = 10000

    OP = MonteCarlo()
    OP.simulations(S0=S0, r=r, sigma=sigma, T=T, steps=steps, I=I)

    # accumulator_price = OP.accumulator(S=S, KO=KO_1, r=r, I=I, n=n, m=m)
    # print(accumulator_price)
    decumulator_price = OP.decumulator(S=S, KO=KO_2, r=r, I=I, n=n, m=m)
    print(decumulator_price)

    # k_out = S * (0.03 + 1)
    # lock_period = 0
    # coupon = 0.2
    # K = 1
    # dn = K * 0.85
    # principal = 1
    
    # snowball_price = OP.snowball(coupon, I, k_out, lock_period, dn)
    # print(snowball_price)





    
