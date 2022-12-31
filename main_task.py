import numpy as np
import math
import time


class Constant:
    pass


class MOD():
    def __init__(self, ml):
        self.ml = ml
        self.nsym = 2 ** ml
        self.rep_b = np.empty((ml, self.nsym), int)
        for idx in range(0, self.nsym):
            for idx_ in range(0, ml):
                self.rep_b[idx_, idx] = (idx >> (self.ml - idx_ - 1)) % 2
        self.w = 2 ** np.arange(ml / 2)[::-1]
        self.w = np.concatenate([self.w, 1j * self.w], axis=0)
        self.val = np.dot(self.w, 2 * self.rep_b - 1)
        self.norm = np.sqrt(3 / (2 * (self.nsym - 1)))
        self.val *= self.norm
        self.lv = 2 ** np.arange(ml)[::-1]
        self.amap = np.array([3, 2, 0, 1, 7, 6, 4, 5, 15, 14, 12, 13, 11, 10, 8, 9], dtype=int)
        self.amap_ = np.array([2, 3, 1, 0, 6, 7, 5, 4, 14, 15, 13, 12, 10, 11, 9, 8], dtype=int)
        # self.x_rep = np.arange(-np.log2(self.nsym)+1,np.log2(self.nsym),2)
        # self.x_rep /= np.sqrt(np.mean(self.x_rep**2)*2)
        # self.x_rep = torch.from_numpy(self.x_rep).float()
        # self.xx_rep = self.x_rep**2

    def demodulation(self, y):
        b_ = np.empty((y.shape[0], self.ml * y.shape[1]), int)
        b_tmp = np.empty((y.shape[1], self.ml), int)
        for idx_k in range(0, y.shape[0]):
            a_ = np.argmin(np.abs(y[idx_k, :] - np.tile(self.val, (y.shape[1], 1)).T) ** 2, axis=0)
            for idx_m in range(0, y.shape[1]):
                b_tmp[idx_m, :] = self.rep_b[:, self.amap_[a_[idx_m]]]
            b_[idx_k, :] = b_tmp.T.reshape(-1)
        return b_


def main_task(param):
    nworker_idx = param[0]  # workerの番号取得
    SIM = param[1]  # SIMの情報を取得
    RES = np.zeros([len(SIM.EsN0), 7])
    TX = Constant()
    CH = Constant()
    err = Constant()
    RX = Constant()
    # 乱数のシードを設定（worker毎に異なる値）
    np.random.seed(1234*nworker_idx)

    # シミュレーション
    for idx_En in range(0, len(SIM.EsN0)):
        SIM.HT.start(nworker_idx, idx_En)

        time.sleep(1)

        CH.N0 = 10**(-SIM.EsN0[idx_En]/10)  # 雑音分散
        CH.sigma = np.sqrt(CH.N0 / 2)
        err.noe = np.zeros([3, 1])
        err.nos = np.zeros([3, 1])
        err.sqerr = 0

        for idx_loop in range(0, math.ceil(SIM.nloop/SIM.nworker)):
            # TX
            mod = MOD(SIM.ml)
            TX.bit = np.random.randint(0, 2, (SIM.Kd, mod.ml))
            TX.bit = TX.bit.T
            TX.alp = np.dot(np.kron(mod.lv, np.eye(1, dtype=int)), TX.bit)
            TX.sym = np.array(mod.val[mod.amap[TX.alp]])
            TX.sig = TX.sym.copy()

            # 雑音生成
            CH.n = CH.sigma*(np.random.randn(1, int(SIM.Kd)) + 1j * np.random.randn(1, int(SIM.Kd)))

            # AWGN
            RX.sig = TX.sig + CH.n

            # 復調器
            RX.bit = mod.demodulation(RX.sig)

            # ERR
            noe_ins = (RX.bit != TX.bit.reshape(-1)).sum()
            err.noe[0] = err.noe[0] + noe_ins
            err.noe[1] = err.noe[1] + int(noe_ins > 0)  # BLER
            # err.noe[1] = err.noe[1] + sum(sum(RX.alp~ = TX.alp)) #SERを求めるために必要
            # err.noe[2] = err.noe[2] + (noe_ins~=0) #FERを求めるのに必要
            err.nos[0] = err.nos[0] + SIM.ml * SIM.Kd
            err.nos[1] = err.nos[1] + 1
            # err.nos[1] = err.nos[1] + SIM.M * SIM.Kd #SERを求めるために必要
            # err.nos[2] = err.nos[2] + 1 #FERを求めるのに必要
            # err.sqerr = err.sqerr + np.mean(mean(abs(CH.H_hat - CH.H). ^ 2)) #MSEを求めるのに必要

        RES[idx_En, :] = np.vstack([err.noe, err.nos, err.sqerr]).reshape(1, -1)
        SIM.HT.finish(nworker_idx, idx_En)
    return RES
