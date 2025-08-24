import numpy as np
from scapy.all import *
import glob
from datetime import datetime, timezone, timedelta
import os
from tqdm import tqdm
import logging
import sys

class Get_IDS2019(): # Interface
    DATE={'0112':'SAT-01-12-2018', '0311':'SAT-03-11-2018'}
    def __init__(self, paths, traffic_type, packet_shape, classType, save_to = None):
        self.paths = paths
        self.save_to = save_to
        self.TRAFFIC_TYPE = traffic_type
        self.IMG_SHAPE = packet_shape
        self.CLASS_TYPE = classType

        if save_to == None:
            raise ValueError("save_to directory must be specified")
        self.directory = save_to

        # b_flows, m_flows, attack_class = self.run(self.date_id,self.data_paths)

    # def get_m_2_list(self, date_id):
    #     m_2_list = None
    #     if date_id == "0311":
    #         m_2_list = [
    #             ("172.16.0.5", "192.168.50.4"), ("192.168.50.4", "172.16.0.5"), 
    #         ]
    #     return m_2_list

    # def get_attack_time_class(self, date_id):
    #     start_end_time = []
    #     grey_area_time = []
    #     attack_class = []
    
    #     if date_id == '0311': 
            
    #         attack_class = ['portmap', 'netbios', 'ldap', 'mssql', 'udp', 'udp-lag', 'syn']
    #         start_end_time = [
    #             (datetime(2018, 11, 3, 10, 1), datetime(2018, 11, 3, 10, 1, 40)),  # Portmap
    #             (datetime(2018, 11, 3, 10, 2, 24), datetime(2018, 11, 3, 10, 3, 58)),  # NetBIOS
    #             (datetime(2018, 11, 3, 10, 21), datetime(2018, 11, 3, 10, 30)),  # LDAP
    #             # (datetime(2018, 11, 3, 10, 34, 2), datetime(2018, 11, 3, 10, 41, 57)),  # MSSQL
    #             (datetime(2018, 11, 3, 10, 34, 59), datetime(2018, 11, 3, 10, 41, 29)),  # MSSQL
    #             (datetime(2018, 11, 3, 10, 53), datetime(2018, 11, 3, 11, 2)),  # UDP
    #             (datetime(2018, 11, 3, 11, 16), datetime(2018, 11, 3, 11, 24)),  # UDP-Lag
    #             (datetime(2018, 11, 3, 11, 29), datetime(2018, 11, 3, 11, 31))   # SYN
    #         ]
    #         grey_area_time = [
    #             (datetime(2018, 11, 3, 9, 43), datetime(2018, 11, 3, 10, 10)),
    #             (datetime(2018, 11, 3, 10, 21), datetime(2018, 11, 3, 10, 31)),
    #             (datetime(2018, 11, 3, 10, 33), datetime(2018, 11, 3, 10, 43)),
    #             (datetime(2018, 11, 3, 10, 53), datetime(2018, 11, 3, 11, 4)),
    #             (datetime(2018, 11, 3, 11, 14), datetime(2018, 11, 3, 11, 25)),
    #             (datetime(2018, 11, 3, 11, 28), datetime(2018, 11, 3, 17, 35)),
    #         ]
    #     return start_end_time, grey_area_time, attack_class

    def save_np_files(self, b_flows, m_flows, attack_class):
        # list to numpy array
        for i in range(len(m_flows)):
            m_flows[i] = np.asarray(m_flows[i])
        b_flows = np.asarray(b_flows)

        # make directories
        directory = self.directory

        if not os.path.exists(directory):
            os.makedirs(directory)

        print(f'Number of each class: ')
        print('-' * 50)

        for i, a in enumerate(attack_class):
            np.save(f'{directory}/{a}_t', m_flows[i], allow_pickle=False)
            print(f'{a} is {len(m_flows[i])}')

        np.save(f'{directory}/benign_t', b_flows, allow_pickle=False)
        print(f'benign is {len(b_flows)}')

    def record_error_pcap(self,path,date_id,type,file_name):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(f'{path}/{date_id}_{type}_errorpcap.txt', 'a') as file:
            file.write(f'{file_name}/n')

    def preprocess_flow(self, pkts):
        pass

    def get_label(self, date_id, pkts, m_2_list, start_end_time, grey_area_time):
        # get information of the first packet
        src_ip = pkts[0].payload.src
        src_port = pkts[0].payload.payload.sport
        dst_ip = pkts[0].payload.dst
        dst_port = pkts[0].payload.payload.dport
        pkt_ip_tuple = np.array([src_ip, dst_ip], dtype='O')

        # 設定 時區
        tz = timezone(timedelta(hours=-3))  # 夏令時間為 -3，其他時間為 -4
        arr_time = datetime.fromtimestamp(int(pkts[0].time), tz)
        arr_time = arr_time.replace(tzinfo=None)  # 移除時區資訊以便後續比較
        # print("arr_time:",arr_time)
        # categorize flows to benign and malicious
        label = 0  # default
        if date_id == "0311":
            for i, t in enumerate(grey_area_time):
                if (t[0] <= arr_time <= t[1]): 
                    label = -1
            for i, t in enumerate(start_end_time):
                if ((m_2_list == pkt_ip_tuple).all(1).any()):
                    if (t[0] <= arr_time <= t[1]):   
                        label = i+1
                        break
     
        return label

    def get_used_pkt(self, pkts):
        if self.TRAFFIC_TYPE == 'TCP':
            pkts = pkts[3:self.IMG_SHAPE[1] + 3]
        else:
            pkts = pkts[:self.IMG_SHAPE[1]]

        return pkts

    def run(self):
        # m_2_list is a list of tuples, which contains all attack-victim combinations(source ip, destination ip).
        m_2_list = None

        # get attack time
        start_end_time, grey_area_time, attack_class = [], [], ["malware"]

        # read pcap dirs
        dirs = self.paths
        b_flows = []
        m_flows = [[] for _ in range(len(attack_class))]
        label_count = [0 for _ in range(len(attack_class) + 1)]

        for d in tqdm(dirs):
            # dirname = d.split('/')[-1]
            # print(f"{d}/*{self.TRAFFIC_TYPE}*")
            files = glob.glob(d)
            # check if d is victim ip
            # if any([(dirname.replace('-','.')).find(p[1]) != 0 for p in m_2_list]):
            if self.CLASS_TYPE == "Malware":
                # print(f"{d}/*{self.TRAFFIC_TYPE}*")
                for f in files:
                    try:
                        pkts = rdpcap(f, count=(self.IMG_SHAPE[1] + 3))
                    except (Scapy_Exception, EOFError):
                        continue

                    # check for skip tcp connection in packet
                    if self.TRAFFIC_TYPE == 'TCP' and len(pkts) < 4:
                        continue
                    label = 1
                    if label >= 0:
                        label_count[label]+=1
                        # if(label==4):
                            # print(f)
                    # print("label:",label)
                    pkts = self.get_used_pkt(pkts)
                    flow = self.preprocess_flow(pkts)
                    if label >= 0:
                        if label == 0:
                            # pass
                            b_flows.append(flow)
                        else:
                            c = label - 1
                            m_flows[c].append(flow)

                    del pkts
            else:
                # print("else")
                for f in files:
                    try:
                        pkts = rdpcap(f, count=(self.IMG_SHAPE[1] + 3))
                    except (Scapy_Exception, EOFError):
                        continue

                    # check for skip tcp connection in packet
                    if self.TRAFFIC_TYPE == 'TCP' and len(pkts) < 4:
                        continue
                    pkts = self.get_used_pkt(pkts)
                    flow = self.preprocess_flow(pkts)
                    b_flows.append(flow)
                    del pkts
            del files
        # save files
        print(label_count)
        self.save_np_files(b_flows, m_flows, attack_class)

class Get_IDS2019_del(Get_IDS2019): #delete IP, Mac, Port
    def __init__(self, paths, traffic_type, packet_shape, classType, save_to = None, version = "victim"):
        if save_to == None:
            raise ValueError("save_to directory must be specified")
        super().__init__(paths, traffic_type, packet_shape, classType, save_to=save_to)

    def preprocess_flow(self, pkts):
        max_size = self.IMG_SHAPE[0]-24
        flow = []
        # for pkt in pkts[3:IMG_SHAPE[1] + 3]:
        for pkt in pkts:  # get the first img_shape[1] packets
            # if Ether not in pkt:
            #     raise Exception("Not Ethernet II")

            # get the first img_shape[0] bytes
            pkt_head = [byte for byte in raw(pkt)]

            pkt_head.extend([0] *max_size)  # padding

            # delete Destination and Source MAC,IP Port
            for start,end in [(0,11),(26,37)] :
                pkt_head[start:end+1] = [None]*(end-start+1)
            pkt_head = [x for x in pkt_head if x is not None]

            flow.extend(pkt_head[:max_size])

        # if the flow has too few packets, padding again
        size = max_size* self.IMG_SHAPE[1]
        if len(flow) < size:
            flow.extend([0] * size)
            flow = flow[:size]
        return flow

def runTCP_del(classType):
    traffic_type = 'TCP'
    packet_shape = (120, 5)
    Get_IDS2019_del(
        paths = glob.glob(f'/sdc1/ytlindata/USTC-TFC2016/split/{classType}/split_*/*{traffic_type}*'),
        traffic_type = traffic_type,
        packet_shape = packet_shape,
        classType = classType,
        save_to = f"/sdc1/ytlindata/USTC-TFC2016/del_{packet_shape[0]}_{packet_shape[1]}_flows(delall)/{classType}/{traffic_type}"
    ).run()

def runUDP_del(classType):
    traffic_type = 'UDP'
    packet_shape = (120, 5)
    Get_IDS2019_del(
        paths = glob.glob(f'/sdc1/ytlindata/USTC-TFC2016/split/{classType}/split_*/*{traffic_type}*'),
        traffic_type = traffic_type,
        packet_shape = packet_shape,
        classType = classType,
        save_to = f"/sdc1/ytlindata/USTC-TFC2016/del_{packet_shape[0]}_{packet_shape[1]}_flows(delall)/{classType}/{traffic_type}"
    ).run()
