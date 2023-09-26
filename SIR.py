import pandas as pd
import numpy as np
import data_processing as dp
import time
import multiprocessing as mp
import os

HOME = os.environ['HOME']



# min_day = 50

# p = 0.3 #0.8
p = 0.8
p_interval = np.linspace( p-0.1, p+0.1, 100 )
# q = 0.01 #0.05
q = 0.05
q_interval = np.linspace( q-0.01, q+0.01, 100 )
min_infected = 50 #20 for p=0.8,q=0.05


def cycle_sim( catagory, n=36000 ):
    print(n)
    data = dp.pre_processing(catagory=catagory)
    record = []

    if catagory == 'sex':
        start_interval = [1,10]
        max_day = 100#300  # infect population larger, spreding time long
        data.rename(columns={'timestep':'timestep', 'female':'source', 'male':'target' },inplace=True)
    elif catagory == 'Bitcoin':
        start_interval = [1, 10]
        max_day = 80#180
        data.rename(columns={'timestep': 'timestep', 'source': 'source', 'target': 'target'}, inplace=True)
    elif catagory == 'Eu':
        start_interval = [1, 3]
        max_day = 10#230
        data.rename(columns={'timestep': 'timestep', 'source': 'source', 'target': 'target'}, inplace=True)
    elif catagory == 'math':
        start_interval = [1, 3]
        max_day = 254
        data.rename(columns={'timestep': 'timestep', 'source': 'source', 'target': 'target'}, inplace=True)
    elif catagory == 'hos':
        start_interval = [1, 5]
        max_day = 70
        data.rename(columns={'timestep': 'timestep', 'source': 'source', 'target': 'target'}, inplace=True)
    elif catagory == 'msg':
        start_interval = [1, 3]
        max_day = 10#47
        data.rename(columns={'timestep': 'timestep', 'source': 'source', 'target': 'target'}, inplace=True)




    data2 = data[ ( data['timestep']<=max_day ) ]#& ( data['timestep']>=min_day )]
    nodes = np.unique( data2[['source','target']].values ) #nodes已经按小到大排好序
    nodes_rename = dict( zip(nodes, range(len(nodes))) )
    data2['source'] = [ nodes_rename[x] for x in data2['source']]
    data2['target'] = [ nodes_rename[x] for x in data2['target']]

    np.random.seed()
    start_scope = np.arange( start_interval[0], start_interval[-1]+1 )
    # e_scope = np.arange(-error,error+1)

    i = 0
    while i < n:
        start = np.random.choice( start_scope ) #from start_interval
        # print(start)

        data3 = data2[(data2['timestep'] >= start) ]

        p_, q_ = np.random.choice( p_interval ), np.random.choice( q_interval )
        # infect, source = simluation( start, data2  )
        infect, recovery, source = simulation( start, data3, p_, q_, catagory=catagory  )

        if len(infect) + len(recovery) > min_infected: #p=0.3,q=0.01,是>15 #p=0.8,q=0.05,是>20
            print('i+r:{0}, infect_num:{1}, recovery_num:{2}, population_size:{3}, catagory:{4}'.
                  format(len(infect) + len(recovery),len(infect),len(recovery),len(nodes), catagory )  )
            # record.append( [t0, e, source, infect, recovery] )
            record.append( [ start, source, infect, recovery] )
            i += 1

    return record





'''generate data'''
def simulation( start, data, p, q, catagory='CollegeMsg' ):

    sources = data[data['timestep'] == start].values #起始天的节点
    sources = sources[:,[0,1]] #[['female','male']]
    #print(len(np.unique(sources)))
    source = np.random.choice( np.unique(sources) ) #随机选择一个作为source

    infect = { source }
    recovery = set()
    # recovery_time = { start+r: infect } #key是恢复时间，value是这个时间恢复的节点

    justice = np.random.random_sample( data.shape[0] ) #和data一样长的一组（0，1）随机数
    data = data[ justice<=p ] #肯定不会传染的记录去掉，剩下的记录，如果某一方有接触，肯定传染。
    # print(justice<=p)
    # print(data.head(5))

    for timestep, temp in data.groupby(['timestep']):
        # print(timestep)
        if catagory == 'sex':
            temp = temp[(temp['source'].isin(infect)) | (temp['target'].isin(infect)) ].values
        elif catagory == 'Bitcoin':
            temp = temp[(temp['source'].isin(infect))].values
        elif catagory == 'Eu':
            temp = temp[(temp['source'].isin(infect))].values
        elif catagory == 'math':
            temp = temp[(temp['source'].isin(infect))].values
        elif catagory == 'msg':
            temp = temp[(temp['source'].isin(infect))].values
        elif catagory == 'hos':
            temp = temp[(temp['source'].isin(infect))].values




        temp = temp[:,[0,1]] #[['female','male']]
        # print( temp )
        new_infect = set(temp.flatten()) - recovery - infect #减去已康复不会再得病的
        # print(infect)

        # justice = np.random.random_sample( len(infect) ) #和infect一样长的一组（0，1）随机数
        # recovery_new = set(np.array(list(infect))[ np.where(justice<=q) ])
        recovery_new = set(filter(lambda x: np.random.random()<=q, infect))
        # recovery_new = recovery_time.get( timestep, set() )
        recovery |= recovery_new
        # print(recovery,recovery_new)
        infect -= recovery_new

        infect |= new_infect  #\
        # recovery_time[timestep+r] = new_infect


    return  infect, recovery, source
    # return infect, source



def parallel( catagory, n =100000, core =12 ):
    pool = mp.Pool( core )
    n_ = int( n/core )
    n = n - n_*core #分成core份后多出来的

    results = []
    for _ in range(core):
        results.append( pool.apply_async( cycle_sim, args=( catagory, n_ ) ) )
    pool.close()
    pool.join()

    record = []
    for i in results:
        record.extend( i.get() )

    record.extend( cycle_sim(catagory, n))

    return  record







if __name__ == '__main__':
    a = time.time()
    # infect, recovery = simluation()
    # print(len(infect),len(recovery))
    # catagory = 'sex'  #
    # catagory = 'Bitcoin'
    # catagory = 'Eu' #'hos'
    # for catagory in [ 'sex', 'Bitcoin', 'Eu', 'hos', 'msg' ] :
    for catagory in [ 'Eu']:
        record = parallel( catagory=catagory, n=20000,core=50 )



        f1 = open(HOME+'/source/data/{0}_reocrd(p={1},q={2}).txt'.format( catagory, p, q ), 'w')
        f1.write(str(record))
        f1.close()

    print(time.time()-a)







