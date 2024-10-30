


import rosbag
import os
import numpy as np
import cv2
from cv_bridge import CvBridge
from scipy import interpolate
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R
import torch
from torch import Tensor
import torchvision
import matplotlib.pyplot as plt

class BagReader:
    def __init__(self,sn:int,base_path,occ_path):
        self.occ_bag_path = f'{base_path}/{occ_path}'
        self.width,self.height = 346,260
        self.bridge = CvBridge()
        self.save_dir = os.path.abspath(f'./sai_result')
        os.makedirs(self.save_dir,exist_ok=True)
        conps = occ_path.split(".")
        self.save_name = f'{conps[0]}'
        self.save_path = os.path.join(self.save_dir,f'{self.save_name}.png')
        self.bad_pixels = np.load(os.path.join(os.path.dirname(__file__),'bad_pixels_new.npy'))
        self.Tx_list = []
        self.Ty_list = []
        self.Tz_list = []
        self.Qx_list = []
        self.Qy_list = []
        self.Qz_list = []
        self.Qw_list = []
        self.quat = None
        self.name_list =[]
        self.fx=755.9776
        self.fy=755.094177
        self.cx=170.413032 
        self.cy=127.521745
        self.K = np.array([[self.fx,0,self.cx],[0,self.fy,self.cy],[0,0,1]])
        self.R_e = None
        self.T_e = None
        
        # q: [-0.03998927 -0.02619826  0.01889107  0.99867794] +- [0.00145186 0.00142233 0.00069052]
	    # t: [-0.13797974  0.05149557 -0.12842144] +- [0.00491838 0.00293956 0.00699347]

    
    def check(self):
        if not os.path.exists(self.occ_bag_path) :
            print(f"No such bag path {self.occ_bag_path}.")
            return False
        if os.path.exists(self.save_path):
            print(f"Skip {self.save_path}.")
            return False
        
        return True
    

    
    def get_pose(self,pose_path):

        file_path = os.path.join(pose_path, 'images.txt')  # 替换为你的文件路径
        with open(file_path, 'r') as file:
            lines = file.readlines()

        for line in lines:
            if '.jpg' in line or '.png'in line :
                data = line.strip().split()
                # 获取最后3个参数
                self.Qw_list.append(np.float64(data[-9]))
                self.Qx_list.append(np.float64(data[-8]))
                self.Qy_list.append(np.float64(data[-7]))
                self.Qz_list.append(np.float64(data[-6]))
                params = np.float64(data[-5:-2])
                self.Tx_list.append(params[0])
                self.Ty_list.append(params[1])
                self.Tz_list.append(params[2])
                self.name_list.append(data[-1].split('_')[0])

        self.name_list=np.array(list(map(int,self.name_list)))

        # 使用zip将多个列表组合成一个元组的列表  
        combined = list(zip(self.name_list,self.Tx_list, self.Ty_list, self.Tz_list,self.Qx_list,self.Qy_list,self.Qz_list,self.Qw_list))  
        
        # 使用sorted函数按name的值进行排序  
        sorted_combined = sorted(combined, key=lambda item: item[0])  
        
        # 使用zip将排序后的元组列表解开成单独的列表  
        self.name_list,self.Tx_list, self.Ty_list, self.Tz_list,self.Qx_list,self.Qy_list,self.Qz_list,self.Qw_list = zip(*sorted_combined)
        
        # # 如果需要，将结果转换为列表  
        self.name_list = list(self.name_list)
        self.Tx_list = list(self.Tx_list)
        self.Ty_list = list(self.Ty_list)
        self.Tz_list = list(self.Tz_list)
        self.Qx_list = list(self.Qx_list)
        self.Qy_list = list(self.Qy_list)
        self.Qz_list = list(self.Qz_list)
        self.Qw_list = list(self.Qw_list)
         

        # 现在sorted_name, sorted_x, sorted_y, sorted_t都是按name排序的

        self.Tx_list= np.array(self.Tx_list)
        self.Ty_list= np.array(self.Ty_list)
        self.Tz_list= np.array(self.Tz_list)
 

        # self.name_list=np.array(list(map(int,self.name_list)))
        self.quat = torch.stack([torch.Tensor(self.Qx_list), torch.Tensor(self.Qy_list), torch.Tensor(self.Qz_list), torch.Tensor(self.Qw_list)], dim=1)

        
        return 0
    
    def rot_interpolator(self,tss_poses_ns,key_rots,tss_query_ns):
        key_rots = R.from_quat(key_rots)
        rot_interpolator = Slerp(tss_poses_ns, key_rots)
        rots = rot_interpolator(tss_query_ns).as_matrix()
        return rots
    
    def gold_search(self,event_x,event_y,event_p,event_t,Xrange,Yrange,frame_sheet):


        fx = self.fx
        fy= self.fy
        w =  346
        h = 260

        x_pose = np.array(self.Tx_list)
        y_pose = np.array(self.Ty_list)
        Tx_max = np.max(x_pose)
        Ty_max = np.max(y_pose)


        if Tx_max > Ty_max:
            d_min = (Tx_max * fx / 2) / (w * 1 / 2)
            d_max = (Tx_max * fx / 2) / (w * 1 / 20)
            print("x is max")   
        else:
            d_min = (Ty_max * fy / 2) / (h * 1 / 2)
            d_max = (Ty_max * fy / 2) / (h * 1 / 20)
            print("y is max")
        #event_t 时间戳 用于插值

        stop_threshold=0.01
        start_depth = d_min
        end_depth = d_max
        delta_depth = (d_max - d_min)*0.618

       
        # Generate etensor
        time_step = frame_sheet
        minT,maxT = event_t.min(),event_t.max()
        event_t_nor = event_t-minT
        interval = (maxT-minT)/time_step

        print(f'sheet={frame_sheet}')
        
        while (delta_depth > stop_threshold):
            fst_focus_d = start_depth + delta_depth / 2
            sec_focus_d = end_depth - delta_depth / 2

            fst_dx = fx*x_pose/fst_focus_d
            sec_dx = fx*x_pose/sec_focus_d
            

            fst_dy = fy*y_pose/fst_focus_d
            sec_dy = fy*y_pose/sec_focus_d


            f_fst = interpolate.interp1d(self.frame_t_list, fst_dx)
            fst_event_dx = f_fst(event_t)
            fst_event_x_focus = event_x - np.round(fst_event_dx)
            fst_event_x_focus = np.clip(fst_event_x_focus, 0, self.width-1)

            f_fst = interpolate.interp1d(self.frame_t_list, fst_dy)
            fst_event_dy = f_fst(event_t)
            fst_event_y_focus = event_y - np.round(fst_event_dy)
            fst_event_y_focus = np.clip(fst_event_y_focus, 0, self.height-1)

            f_sec = interpolate.interp1d(self.frame_t_list, sec_dx)
            sec_event_dx = f_sec(event_t)
            sec_event_x_focus = event_x - np.round(sec_event_dx)
            sec_event_x_focus = np.clip(sec_event_x_focus, 0, self.width-1)

            f_sec = interpolate.interp1d(self.frame_t_list, sec_dy)
            sec_event_dy = f_sec(event_t)
            sec_event_y_focus = event_y - np.round(sec_event_dy)
            sec_event_y_focus = np.clip(sec_event_y_focus, 0, self.height-1)

        

            # convert events to event tensors
            pos = np.zeros((time_step, self.height, self.width))
            neg = np.zeros((time_step, self.height, self.width))
            T,H,W = pos.shape
            pos,neg = pos.ravel(),neg.ravel()
            ind = (event_t_nor/interval).astype(int)
            ind[ind==T] -= 1
            fst_event_x_focus,fst_event_y_focus = fst_event_x_focus.astype(int),fst_event_y_focus.astype(int)
            pos_ind,neg_ind = event_p == 1,event_p == 0
            np.add.at(pos, fst_event_x_focus[pos_ind] + fst_event_y_focus[pos_ind]*W + ind[pos_ind]*W*H, 1)
            np.add.at(neg, fst_event_x_focus[neg_ind] + fst_event_y_focus[neg_ind]*W + ind[neg_ind]*W*H, 1)
            pos = np.reshape(pos, (T,H,W))
            neg = np.reshape(neg, (T,H,W))
            pos = pos[:, Yrange[0]:Yrange[1], Xrange[0]:Xrange[1]]
            neg = neg[:, Yrange[0]:Yrange[1], Xrange[0]:Xrange[1]]

            fst_sum_pos = np.sum(pos+neg,axis=0)
            fst_event_var = torch.var(torch.from_numpy(fst_sum_pos))
            del pos,neg

            pos = np.zeros((time_step, self.height, self.width))
            neg = np.zeros((time_step, self.height, self.width))
            T,H,W = pos.shape
            pos,neg = pos.ravel(),neg.ravel()
            ind = (event_t_nor/interval).astype(int)
            ind[ind==T] -= 1
            sec_event_x_focus,sec_event_y_focus = sec_event_x_focus.astype(int),sec_event_y_focus.astype(int)
            pos_ind,neg_ind = event_p == 1,event_p == 0
            np.add.at(pos, sec_event_x_focus[pos_ind] + sec_event_y_focus[pos_ind]*W + ind[pos_ind]*W*H, 1)
            np.add.at(neg, sec_event_x_focus[neg_ind] + sec_event_y_focus[neg_ind]*W + ind[neg_ind]*W*H, 1)
            pos = np.reshape(pos, (T,H,W))
            neg = np.reshape(neg, (T,H,W))
            pos = pos[:, Yrange[0]:Yrange[1], Xrange[0]:Xrange[1]]
            neg = neg[:, Yrange[0]:Yrange[1], Xrange[0]:Xrange[1]]
            sec_sum_pos = np.sum(pos+neg,axis=0)
            sec_event_var = torch.var(torch.from_numpy(sec_sum_pos))
            del pos,neg

            if(fst_event_var>sec_event_var):
                start_depth = start_depth
                end_depth = start_depth + delta_depth
            else:
                start_depth = end_depth - delta_depth
                end_depth = end_depth
            delta_depth = (end_depth - start_depth) * 0.618

        best_focus_depth = (start_depth+end_depth)/2
        print(f'best_focus_depth={best_focus_depth}')

        best_dx = fx*x_pose/best_focus_depth
        best_dy = fy*y_pose/best_focus_depth

        f_best = interpolate.interp1d(np.array(self.frame_t_list), best_dx)
        best_event_dx = f_best(event_t)
        best_event_x_focus = event_x - np.round(best_event_dx)
        best_event_x_focus = np.clip(best_event_x_focus, 0, self.width-1)
        
        f_best = interpolate.interp1d(np.array(self.frame_t_list), best_dy)
        best_event_dy = f_best(event_t)
        best_event_y_focus = event_y - np.round(best_event_dy)
        best_event_y_focus = np.clip(best_event_y_focus, 0, self.height-1)

        
        
        pos = np.zeros((time_step, self.height, self.width))
        neg = np.zeros((time_step, self.height, self.width))

        T,H,W = pos.shape
        pos,neg = pos.ravel(),neg.ravel()
        ind = (event_t_nor/interval).astype(int)
        ind[ind==T] -= 1
        best_event_x_focus,best_event_y_focus = best_event_x_focus.astype(int),best_event_y_focus.astype(int)
        pos_ind,neg_ind = event_p == 1,event_p == 0
        np.add.at(pos, best_event_x_focus[pos_ind] + best_event_y_focus[pos_ind]*W + ind[pos_ind]*W*H, 1)
        np.add.at(neg, best_event_x_focus[neg_ind] + best_event_y_focus[neg_ind]*W + ind[neg_ind]*W*H, 1)
        pos = np.reshape(pos, (T,H,W))
        neg = np.reshape(neg, (T,H,W))
        # all_sum_pos = np.sum(pos+neg,axis=0)
        pos = pos[:, Yrange[0]:Yrange[1], Xrange[0]:Xrange[1]]
        neg = neg[:, Yrange[0]:Yrange[1], Xrange[0]:Xrange[1]]
        best_sum_pos = np.sum(pos+neg,axis=0)

        best_sum_pos[best_sum_pos>3*np.mean(best_sum_pos)]=3*np.mean(best_sum_pos)
        best_sum_pos/= np.max(best_sum_pos)
        best_sum_pos = best_sum_pos*255

        return best_sum_pos , best_focus_depth
        
        
    def readbag(self,sn:int,occ_name:str):
        self.event_list = []
        self.frame_list = []
        self.frame_t_list = []
        self.occ_bag = rosbag.Bag(self.occ_bag_path)
        # read occ bag
        for topic, msgs, t in self.occ_bag.read_messages():
            if topic == '/dvs/events':
                self.event_list += msgs.events
                # print(len(msgs.events))
            if topic == '/camera/image_color/compressed':
                # image = self.bridge.imgmsg_to_cv2(msgs,"bgr8")
                # self.frame_list.append(image)

                tss = msgs.header.stamp.secs+msgs.header.stamp.nsecs*1e-9
                self.frame_t_list.append(tss)
    
        event_x = np.array(list(map(lambda x: x.x, self.event_list)))
        event_y = np.array(list(map(lambda x: x.y, self.event_list)))
        event_t = np.array(list(map(lambda x: x.ts.secs + x.ts.nsecs*1e-9, self.event_list)))
        event_p = np.array(list(map(lambda x: x.polarity, self.event_list)))
        for y,x in self.bad_pixels:
            index_x = np.where(event_x == x)
            index_y = np.where(event_y == y)
            index = np.intersect1d(index_x,index_y)
            event_x = np.delete(event_x,index)
            event_y = np.delete(event_y,index)
            event_t = np.delete(event_t,index)
            event_p = np.delete(event_p,index)
        

        #get_pose
        flie_path=os.path.join('..','img',self.save_name)
        print("flie_path:",flie_path)
        self.get_pose(flie_path)
        
        self.frame_t_list=np.array(self.frame_t_list)
        print(f'self.t_list.shape={self.frame_t_list.shape}')
        self.frame_t_list=self.frame_t_list[self.name_list]
        print(f'self.t_list.shape={self.frame_t_list.shape}')
       
        time_range = [self.frame_t_list[0],self.frame_t_list[-1]]
        # event 去除 超出时间范围的 点
        event_index = (event_t>time_range[0]) & (event_t < time_range[-1])
        event_x = event_x[event_index]
        event_y = event_y[event_index]
        event_t = event_t[event_index]
        event_p = event_p[event_index]

        # res_x,res_y = 256,256 # roi
        # Xrange = ((self.width-res_x)//2,(self.width-res_x)//2+res_x)
        # Yrange = ((self.height-res_y)//2,(self.height-res_y)//2+res_y)
        # Xrange = (0,346)
        # Yrange = (0,260)

        r_martix = self.rot_interpolator(self.frame_t_list,self.quat,event_t)
        print(f'r_martix.shape={r_martix.shape}')


        '''
        所以公式为
        # xref = KRK(-1)x_i - KT/d '''
    #      q: [-0.02943567 -0.04885097 -0.02404991  0.99808252] +- [0.05281802 0.01019196 0.04928858]
	#  t: [-2.91456305 -0.27014638 -5.37570435] +- [0.03146328 0.25745342 0.03648948]

        # self.T_e = np.array([-2.91456305,-0.27014638,-5.37570435])
        # self.R_e = np.array([-0.02943567,-0.04885097,-0.02404991,0.99808252])
        # self.R_e = R.from_quat(self.R_e)


        # 计算R位移
        # r_martix = np.transpose(r_martix,(0,2,1))
        K_inv = np.linalg.inv(self.K)
        r_move = self.K@r_martix
        r_move = r_move@K_inv-np.eye(3)

        z= np.ones_like(event_y)
        pix_temp = np.stack([event_x,event_y,z],axis=0)
        pix_temp = np.stack([pix_temp[:,x].reshape(3,1) for x in range(pix_temp.shape[1])],axis=0)

        r_move_pix = r_move@pix_temp

        print(f'r_move_pix.shape={r_move_pix.shape}')
        # 计算t位移
        f_x = interpolate.interp1d(np.array(self.frame_t_list), self.Tx_list)
        f_y = interpolate.interp1d(np.array(self.frame_t_list), self.Ty_list)
        f_z = interpolate.interp1d(np.array(self.frame_t_list), self.Tz_list)
        print(f'self.Tx_list.shape={self.Tx_list.shape}')

        T_x = f_x(event_t)
        T_y = f_y(event_t)
        T_z = f_z(event_t)


        
        T_matrix = np.stack([T_x,T_y,T_z],axis=0)
        swapped_matrix = T_matrix.swapaxes(0, 1)
        T = swapped_matrix.reshape(-1, 3, 1)
        T_move = self.K@T
        print(f'T_move.shape={T_move.shape}')

        # 计算 压帧数
        time_step = len(self.frame_t_list)
        minT,maxT = event_t.min(),event_t.max()
        event_t_nor = event_t-minT
        interval = (maxT-minT)/time_step

    

        for best_focus_depth in np.arange(60,90,1):
            pos = np.zeros((time_step, self.height, self.width))
            neg = np.zeros((time_step, self.height, self.width))
            T_move_d = T_move/best_focus_depth
            x_move = np.array([ r_move_pix[i,1,0] +  T_move_d[i,0,0]  for i in range(len(event_t))])
            y_move = np.array([ r_move_pix[i,1,0] +  T_move_d[i,1,0]  for i in range(len(event_t))])

            best_event_x_focus = event_x - np.round(x_move)
            best_event_x_focus = np.clip(best_event_x_focus, 0, self.width-1)

            best_event_y_focus = event_y - np.round(y_move)
            best_event_y_focus = np.clip(best_event_y_focus, 0, self.height-1)

            T,H,W = pos.shape
            pos,neg = pos.ravel(),neg.ravel()
            ind = (event_t_nor/interval).astype(int)
            ind[ind==T] -= 1
            best_event_x_focus,best_event_y_focus = best_event_x_focus.astype(int),best_event_y_focus.astype(int)
            pos_ind,neg_ind = event_p == 1,event_p == 0
            np.add.at(pos, best_event_x_focus[pos_ind] + best_event_y_focus[pos_ind]*W + ind[pos_ind]*W*H, 1)
            np.add.at(neg, best_event_x_focus[neg_ind] + best_event_y_focus[neg_ind]*W + ind[neg_ind]*W*H, 1)
            pos = np.reshape(pos, (T,H,W))
            neg = np.reshape(neg, (T,H,W))
            best_sum_pos = np.sum(pos+neg,axis=0)
            best_sum_pos[best_sum_pos>3*np.mean(best_sum_pos)]=3*np.mean(best_sum_pos)
            best_sum_pos/= np.max(best_sum_pos)
            best_sum_pos = best_sum_pos*255
            os.makedirs(os.path.join(".","sai_result",self.save_name),exist_ok=True)
            cv2.imwrite(os.path.abspath(f'./sai_result/{self.save_name}/event_{best_focus_depth:.2f}.png'), best_sum_pos)
            del pos,neg
     

        # img , best_depth = self.gold_search(event_x,event_y,event_p,event_t,Xrange,Yrange,frame_sheet=len(self.frame_t_list))

        # os.makedirs(os.path.join(".","sai_result",self.save_name),exist_ok=True)
        # cv2.imwrite(os.path.abspath(f'./sai_result/{self.save_name}/{best_depth}.png'), img)



    
   

if __name__ == "__main__":
    base_path = os.path.abspath('../bag5')
    # fx,v = 485.,0.1775
    bag_list = os.listdir(base_path)
    bag_list = sorted(filter(lambda x:"003" in x,bag_list))
    print(bag_list)




    reader = BagReader(1,base_path,bag_list[-1])
    occ_name = bag_list[-1].split(".")[0]
    if not os.path.exists(os.path.join(".","sai_result",occ_name)) :
            os.makedirs(os.path.join(".","sai_result",occ_name),exist_ok=True)

    if reader.check() :
        print(occ_name)
        reader.readbag(sn=1,occ_name=occ_name)
    # for sn,occ_path in enumerate(bag_list):
    #     reader = BagReader(sn,base_path,occ_path)
    #     occ_path = occ_path.split(".")[0]
    #     if not os.path.exists(os.path.join(".","events_acc",occ_path)) :
    #         os.makedirs(os.path.join(".","events_acc",occ_path),exist_ok=True)
    #     if reader.check() :
    #         print(occ_path)
    #         reader.readbag(sn=sn)
