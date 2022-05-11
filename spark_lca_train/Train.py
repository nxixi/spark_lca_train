import pandas as pd
import numpy as np
from statsmodels.tsa import stattools


class Train(object):

    @staticmethod
    def from_map(conf_dict):
        agg = conf_dict.get("agg", 5)
        customAccidentalParams = conf_dict.get("customAccidentalParams")
        generalAccidentalParams = conf_dict.get("generalAccidentalParams")
        rolling_hours = conf_dict.get("rolling_hours", 12)
        rolling_percent = conf_dict.get("rolling_percent", 95)
        his_window = conf_dict.get("his_window", 15)
        acc_thresh = conf_dict.get("acc_thresh", 0.5)

        train = Train(agg=agg, customAccidentalParams=customAccidentalParams, generalAccidentalParams=generalAccidentalParams,
                      rolling_hours=rolling_hours, rolling_percent=rolling_percent, his_window=his_window, acc_thresh=acc_thresh)

        return train

    def __init__(self, agg, customAccidentalParams, generalAccidentalParams, rolling_hours=12, rolling_percent=95,
                 his_window=15, acc_thresh=0.5):
        self.agg = agg
        self.customAccidentalParams = customAccidentalParams   # 自定义偶发配置
        self.generalAccidentalParams = generalAccidentalParams   # 通用偶发配置
        self.rolling_hours = rolling_hours  # 判断稀疏性时滑动窗口大小，单位小时
        self.rolling_percent = rolling_percent   # 判断稀疏性时滑动平均数的分位点
        self.his_window = his_window
        self.acc_thresh = acc_thresh

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
    def cal_is_accidental(self, tem_id, count, total_count):
        acci_count_thresh = self.customAccidentalParams[tem_id]['acci_count_thresh'] if tem_id in self.customAccidentalParams else \
        self.generalAccidentalParams['acci_count_thresh']
        acci_occu_thresh = self.customAccidentalParams[tem_id]['acci_occu_thresh'] if tem_id in self.customAccidentalParams else \
        self.generalAccidentalParams['acci_occu_thresh']

        if total_count == 0:
            return False
        elif (count < acci_count_thresh) and (count / total_count < 0.01 * acci_occu_thresh):
            return True
        else:
            return False

    # 检验稀疏性、周期性
    def run(self, value_his):
        # 计算每个模板、所有模板近acci_day天发生的日志总量
        count_dict = {}
        total_count = 0
        for tem_id in value_his:
            acci_day = self.customAccidentalParams[tem_id]['acci_day'] if tem_id in self.customAccidentalParams else self.generalAccidentalParams['acci_day']
            count = sum(value_his[tem_id][-self.list_len_one_hour * 24 * acci_day:])
            count_dict[tem_id] = count
            total_count += count
        # 每个模板的规律性和偶发性
        reg = {}
        for tem_id in value_his:
            history_value = value_his[tem_id]
            if len(history_value) > self.list_len_one_hour * 24:
                is_sparse = self.sparse(history_value)
                is_period, per_cor, per_coef = self.period(history_value)
            else:
                is_sparse, is_period, per_cor, per_coef = False, False, None, None
            is_accidental = self.cal_is_accidental(tem_id, count_dict[tem_id], total_count)
            reg[tem_id] = {'value_his': value_his[tem_id], 'is_sparse': is_sparse, 'is_period': is_period,
                           'per_cor': per_cor, 'per_coef': per_coef, 'is_accidental': is_accidental}
        return reg