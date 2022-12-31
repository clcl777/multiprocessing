import multiprocessing
import psutil
import datetime
import time
import numpy as np


class HPCToolkit:
    def __init__(self, SIM: object):
        self.path = 'progress_bar.txt'
        self.cpu_count = get_spec1()
        # get_spec2()
        dt_now = datetime.datetime.now()
        self.start_time_month = str(dt_now.month)
        self.start_time_day = str(dt_now.day)
        self.start_time_hour = str(dt_now.hour)
        self.start_time_minute = str(dt_now.minute)
        if dt_now.minute < 10:
            self.start_time_minute = "0" + str(dt_now.minute)
        self.start_time = self.start_time_month + "/" + self.start_time_day + " " + self.start_time_hour + ":" + self.start_time_minute
        self.nworker = SIM.nworker
        self.nloop = SIM.nloop
        self.EsN0 = SIM.EsN0
        # self.process_start_time_list = [['' for i in range(EsN0.shape[0])] for j in range(nworker)]
        # self.process_finish_time_list = [['' for i in range(EsN0.shape[0])] for j in range(nworker)]
        self.process_start_time_ndarray = np.zeros((self.nworker, self.EsN0.shape[0]))
        self.process_finish_time_ndarray = np.zeros((self.nworker, self.EsN0.shape[0]))
        self.lock = SIM.lock
        # 初期値書き込み
        self.write_initial(self.get_str_spec(self.nworker))
        print()

    def start(self, process_idx: int, EsN0_idx: int):
        # self.process_start_time_list[process_idx][EsN0_idx] = time.time()
        self.process_start_time_ndarray[process_idx, EsN0_idx] = time.time()

    def finish(self, process_idx: int, EsN0_idx: int):
        # self.process_finish_time_list[process_idx][EsN0_idx] = time.time()
        self.process_finish_time_ndarray[process_idx, EsN0_idx] = time.time()
        average_time = np.mean(self.process_finish_time_ndarray[process_idx, :] - self.process_start_time_ndarray[process_idx, :])
        d, h, m, s = get_d_h_m_s(float(average_time))
        average_time_text = str(d) + "d " + str(h) + "h " + str(m) + "m " + str(s) + "s" + "/iter"
        text = str(process_idx) + " " + progressbar(EsN0_idx, self.EsN0.shape[0]-1) + " " + average_time_text + "\n"
        self.write_finish_process(process_idx, text)

    def write_initial(self, text: str):
        with open(self.path, mode='w') as f:
            f.write(text)

    def write_finish_process(self, process_idx: int, string: str):
        self.lock.acquire()
        with open(self.path) as file:
            text_list = file.readlines()
        text_list[process_idx + 5] = string
        with open(self.path, mode='w') as f:
            f.writelines(text_list)
        self.lock.release()

    def get_str_spec(self, nwoker) -> str:
        start_text = "Start: " + self.start_time
        expect_text = "Expect: "
        cpu_count_text = "Active core: " + str(self.cpu_count)
        nwoker_text = "Process: " + str(nwoker)
        progress_bar_text = ""
        for i in range(self.nworker):
            progress_bar_text += str(i) + " " + progressbar(0, self.EsN0.shape[0]) + "\n"
        return start_text + "\n" + expect_text + "\n" + cpu_count_text + "\n" + nwoker_text + "\n\n" + progress_bar_text


def get_spec1():
    cpu_count = multiprocessing.cpu_count()  # 論理コアの数
    return cpu_count


def get_spec2():
    cpu_count_physical = psutil.cpu_count(logical=False)
    cpu_count_logical = psutil.cpu_count(logical=True)
    total_memory_GB = psutil.virtual_memory().total/(1024**3)
    available_memory_GB = psutil.virtual_memory().available/(1024**3)
    cpu_core_usage = psutil.cpu_percent(interval=1, percpu=True)
    # psutil.sensors_temperatures()['coretemp']
    return cpu_count_physical, cpu_count_logical, total_memory_GB, available_memory_GB, cpu_core_usage


def progressbar(current, max) -> str:
    # args
    #     current: int/float 現在値
    #     max: int/float 最大値
    # length:いじれる。表示したい長さに合わせる。
    # bar:いじれる。好みでどうぞ。今見えてるやつは、プロポーショナルフォントでも使える
    ratio = current / max
    length = 20
    progress = int(ratio * length)
    bar = f'[{"■" * progress}{"□" * (length - progress)}]'
    percentage = int(ratio * 100)
    return f'{bar} {percentage}%'


def get_d_h_m_s(sec: float):
    td = datetime.timedelta(seconds=sec)
    m, s = divmod(td.seconds, 60)
    h, m = divmod(m, 60)
    d = td.days
    return d, h, m, s


def main():
    # path = 'progress_bar.txt'
    # s = 'New file2'
    # cpu_count = get_spec1()

    # with open(path, mode='w') as f:
    #     f.write(s)
    EsN0 = np.arange(0, 10, 2)
    HT = HPCToolkit(4, 10, EsN0)


if __name__ == "__main__":
    main()
