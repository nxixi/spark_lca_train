import pandas as pd
import numpy as np
from statsmodels.tsa import stattools


class Train(object):

    @staticmethod
    def from_map(conf_dict):
        agg = conf_dict.get("agg", 5)
        is_mode_all = conf_dict.get("is_mode_all", True)
        rolling_hours = conf_dict.get("rolling_hours", 12)
        rolling_percent = conf_dict.get("rolling_percent", 95)
        his_window = conf_dict.get("his_window", 15)
        acc_thresh = conf_dict.get("acc_thresh", 0.5)
        acci_occu_thresh = conf_dict.get("acci_occu_thresh", 2)
        acci_count_thresh = conf_dict.get("acci_count_thresh", 10)

        train = Train(agg=agg, is_mode_all=is_mode_all, rolling_hours=rolling_hours, rolling_percent=rolling_percent,
                      his_window=his_window, acc_thresh=acc_thresh, acci_occu_thresh=acci_occu_thresh, acci_count_thresh=acci_count_thresh)

        return train

    def __init__(self, agg, is_mode_all=True, rolling_hours=12, rolling_percent=95, his_window=15, acc_thresh=0.5,
                 acci_occu_thresh=2, acci_count_thresh=10):
        self.agg = agg
        self.is_mode_all = is_mode_all   # 是否训练所有模板，否则训练一个模板
        self.rolling_hours = rolling_hours  # 判断稀疏性时滑动窗口大小，单位小时
        self.rolling_percent = rolling_percent   # 判断稀疏性时滑动平均数的分位点
        self.his_window = his_window
        self.acc_thresh = acc_thresh
        self.acci_occu_thresh = acci_occu_thresh   # 偶发判断百分比阈值
        self.acci_count_thresh = acci_count_thresh   # 偶发判断固定阈值

        self.is_condition = False   # 是否满足规律检测条件（历史数据量大于一天）
        self.list_len_one_hour = int(60 / self.agg)  # 一小时有多少个点

    # 判断是否稀疏
    def sparse(self, history_value):
        smooth_median = pd.Series(history_value).rolling(window=self.list_len_one_hour * self.rolling_hours).median()
        value = np.percentile(smooth_median.dropna(), self.rolling_percent)
        if value > 0:
            return False
        else:
            return True

    # 判断是否周期
    def period(self, history_value):
        start_loc = self.list_len_one_hour  # 检测1小时以上的周期
        acf = stattools.acf(history_value, nlags=self.list_len_one_hour * 24 * self.his_window)
        cor_ind = np.argmax(acf[start_loc:]) + start_loc
        max_th = cor_ind + self.list_len_one_hour // 2
        if max_th > len(acf) - 1:
            max_th = len(acf) - 1
        if acf[cor_ind] > self.acc_thresh and (
                acf[cor_ind] > acf[cor_ind - self.list_len_one_hour // 2] and acf[cor_ind] > acf[max_th]):
            return True, cor_ind, (1 - acf[cor_ind]) * 2
        else:
            return False, None, None

    # 判断是否偶发
    def cal_is_accidental(self, count, total_count):
        if (count < self.acci_count_thresh) and (count / total_count < 0.01 * self.acci_occu_thresh):
            return True
        else:
            return False

    # def process_history_value(self, last_time, history_value):
    #     last_time_split = last_time.split(' ')
    #     train_time = last_time_split[0] + ' ' + self.train_time
    #     time_struct = datetime.strptime(train_time, "%Y-%m-%d %H:%M:%S")
    #     diff_length_ = (int((time_struct - datetime.strptime(last_time, "%Y-%m-%d %H:%M:%S")).total_seconds() // (
    #                 self.agg * 60)) - 1)
    #     if self.train_time > last_time_split[1]:
    #         diff_length = diff_length_
    #     else:
    #         diff_length = int(1440 / self.agg) - diff_length_
    #     history_value += [0] * diff_length
    #     history_value = history_value[-min(self.list_len_one_hour * 24 * self.his_window, len(history_value)):]
    #     return history_value

    # 检验稀疏性、周期性
    def run(self, value_his, acci_dict, total_count, template_id, reg):
        # 稀疏性和周期性检测、偶发检测
        if self.is_mode_all:
            for tem_id in value_his:
                history_value = value_his[tem_id]
                if len(history_value) > self.list_len_one_hour * 24:
                    # history_value = self.process_history_value(last_time, history_value)
                    is_sparse = self.sparse(history_value)
                    is_period, per_cor, per_coef = self.period(history_value)
                else:
                    is_sparse, is_period, per_cor, per_coef = False, False, None, None
                is_accidental = self.cal_is_accidental(acci_dict[tem_id], total_count)
                reg[tem_id] = {'is_sparse': is_sparse, 'is_period': is_period, 'per_cor': per_cor, 'per_coef': per_coef,
                               'is_accidental': is_accidental}
        else:    # TODO
            history_value = value_his[template_id] if template_id in value_his else []
            if len(history_value) > self.list_len_one_hour * 24:
                # history_value = self.process_history_value(last_time, history_value)
                is_sparse = self.sparse(history_value)
                is_period, per_cor, per_coef = self.period(history_value)
            else:
                is_sparse, is_period, per_cor, per_coef = False, False, None, None
            is_accidental = self.cal_is_accidental(acci_dict[template_id], total_count)
            reg[template_id] = {'is_sparse': is_sparse, 'is_period': is_period, 'per_cor': per_cor, 'per_coef': per_coef,
                                'is_accidental': is_accidental}
