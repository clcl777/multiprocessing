import numpy as np
import main_task
import plot_ber
import multiprocessing
from scipy import special
import hpctoolkit


class Constant:
    pass


if __name__ == '__main__':
    SIM = Constant()
    SIM.nworker = 19
    SIM.Kd = 32
    SIM.wloop = 3
    SIM.ml = 4
    SIM.EsN0 = np.arange(0, 11, 2)
    SIM.nloop = 10**SIM.wloop
    SIM.Q = 2**SIM.ml
    RES = np.zeros([len(SIM.EsN0), 7])
    manager = multiprocessing.Manager()
    SIM.lock = manager.Lock()

    SIM.HT = hpctoolkit.HPCToolkit(SIM)

    if SIM.nworker == 1:
        RES = main_task.main_task([0, SIM])
    else:
        # 並列処理
        pool = multiprocessing.Pool(processes=SIM.nworker)
        params = []
        for p in range(SIM.nworker):
            params.append([p, SIM])  # wokerの番号を渡す
        RES_ = pool.map(main_task.main_task, params)
        pool.close()
        RES = sum(RES_)

    SIM.BER = RES[:, 0]/RES[:, 3]
    SIM.BLER = RES[:, 1] / RES[:, 4]
    gamma = 10**(SIM.EsN0/10)
    SIM.BER_theoretical = 3/8*special.erfc(np.sqrt(gamma/10))
    plot_ber.plot_ber(SIM)
