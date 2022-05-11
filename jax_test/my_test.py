import pandas as pd
import time
import datetime
from spark_lca_train.LcaTrain import *
import pickle


def time2stamp(tim):
    timeArray = time.strptime(str(tim), "%Y-%m-%d %H:%M:%S")
    timeStamp = int(time.mktime(timeArray))
    return timeStamp * 1000


def stamp2time(stamp):  # 时间转换：1602259200 -> '2020-10-10 00:00:00'
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stamp))


def start_train(value_his, end_day):
    lca_train = LcaTrain()
    lca_train.configure({'customAccidentalParams': {},
                         'generalAccidentalParams': {'acci_day': 7, 'acci_count_thresh': 10, 'acci_occu_thresh': 2}})
    result = lca_train.transform(value_his)

    with open('out/' + str(end_day) + '.pkl', 'wb') as f:
        pickle.dump(result, f)


if __name__ == '__main__':

    data = pd.read_csv('test_data/my_test_data.csv')
    data['day'] = data['@timestamp'].apply(lambda x: pd.to_datetime(x).date())
    data = data.sort_values(by='@timestamp')

    days = []
    start_day = pd.to_datetime(data['day'].iloc[0]).date()
    end_day = pd.to_datetime(data['day'].iloc[-1]).date()
    this_day = start_day
    while this_day <= end_day:
        days.append(this_day)
        this_day = this_day + datetime.timedelta(days=1)
    for i in days:
        dat = data[data['day'] <= i]
        value_his = {}
        for key, group in dat.groupby('template_id'):
            end = str(i) + ' 23:55:00'
            group = group.append(pd.DataFrame(
                {'@value': [0], 'template_id': [key], '@timestamp': [end], 'time': time2stamp(end),
                 'day': group.iloc[-1]['day']}))
            group.index = pd.to_datetime(group['@timestamp'])
            this_dat = pd.DataFrame()
            this_dat['@value'] = group['@value'].resample('5min').sum()
            this_dat['template_id'] = key
            this_dat['@timestamp'] = this_dat.index
            this_dat['time'] = this_dat['@timestamp'].apply(lambda x: time2stamp(str(x)))
            if len(this_dat) >= 288:
            # if (this_dat['time'].iloc[-1] - this_dat['time'].iloc[0]) // 86400000 >= 2:
                value_his[key] = list(this_dat['@value'])[-288*15:]
        if value_his:
            start_train(value_his, end_day=i)
