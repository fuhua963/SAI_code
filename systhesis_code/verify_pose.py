# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.trans


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
        # T = np.stack([T_matrix[:,x].reshape(3,1) for x in range(T_matrix.shape[1])],axis=0)
        # T =  r_martix@T
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
form import Slerp
import cv2
import os
import torch
from torch import Tensor
import torchvision
import torchvision.transforms.functional as F
from torchvision import transforms
from PIL import Image
from skimage import data,exposure
import re
# 自适应直方图均衡化  AHE



                       



def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()

def bin2depth(i, depth_map, depthdir):

    depth_map = read_array(depth_map)

    min_depth, max_depth = np.percentile(depth_map[depth_map>0], [min_depth_percentile, max_depth_percentile])
    # print(f"min_depth={min_depth}, max_depth={max_depth}")
    # depth_map[depth_map <= 0] = 0   # 把0和负数都设置为nan，防止被min_depth取代
    depth_map[depth_map < min_depth] = min_depth
    depth_map[depth_map > max_depth] = max_depth
    # depth_map = np.nan_to_num(depth_map) # nan全都变为0
    # print(f"max_depth={depth_map.max()}, min_depth={depth_map.min()}")

    threshold_depth  = (max_depth-min_depth)*0.70+min_depth
    # print(threshold_depth)
    depth_map[depth_map <= threshold_depth] = 0
    depth_map[depth_map >= threshold_depth] = 1
    depth_map = depth_map.astype(int)

    return depth_map


def extract_number(filename):
    match = re.search(r'\((\d+)\)', filename)
    if match:
        return int(match.group(1))
    return None



def dcn_warp(voxelgrid: Tensor, flow_x: Tensor, flow_y: Tensor):
    # voxelgrid: [bs,ts,H,W] | flow: [bs,ts,H,W]
    bs, ts, H, W = voxelgrid.shape
    flow = torch.stack([flow_y, flow_x], dim=2)  # [bs,ts,2,H,W]
    flow = flow.reshape(bs, ts * 2, H, W)  # [bs,ts*2,H,W]
    #  单位矩阵 保证了 只对一张图处理
    weight = torch.eye(ts, device=flow.device).double().reshape(ts, ts, 1, 1)  # 返回 ts 张图 对ts张图做处理
    # 单位卷积核
    return torchvision.ops.deform_conv2d(voxelgrid, flow, weight)  # [bs,ts,H,W]


def nonzero_mean(tensor: Tensor, refoucs_mask:Tensor,dim: int, keepdim: bool = False, eps=1e-3):
    numel =torch.squeeze(torch.sum(refoucs_mask, dim=dim, keepdim=keepdim))

    value =torch.squeeze(torch.sum(tensor, dim=dim, keepdim=keepdim))
    numel = torch.unsqueeze(numel, -1)
    return value / (numel + eps)


def read_images(directory_path, image_size):
    # 获取目录中所有图像文件的文件名
    image_files = [f for f in os.listdir(directory_path) if f.endswith('.jpg') or f.endswith('.png')]
    image_files = sorted(image_files,key=lambda x :  int(x.split('_')[0]))
    global N
    N = len(image_files)
    # 初始化矩阵
    images_matrix = np.zeros((len(image_files), *image_size), dtype=np.uint8)
    image_name_list=[]
    # 读取每张图像并存储到矩阵中
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(directory_path, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        images_matrix[i] = np.array(image)

    return images_matrix,image_name_list


def pix_move(width,height,quat,depth,K,T_x,T_y,T_z):

    T_matrix = np.stack([T_x,T_y,T_z],axis=1)

    T = T_matrix.reshape(-1, 3, 1)  # 116,3,1

    quat = np.array(quat)

    pix_x = np.arange(0,width)
    pix_y = np.arange(0,height)
    pix_x , pix_y = np.meshgrid(pix_x,pix_y)
    pix_x = pix_x.reshape(-1)
    pix_y = pix_y.reshape(-1)
    print(pix_x)
    ones = np.ones_like(pix_x)

    # 旋转矩阵 计算
    R_matrix = R.from_quat(quat).as_matrix()  # 116，3，3
    K_inv = np.linalg.inv(K)
    K = K[np.newaxis,:,:].repeat(R_matrix.shape[0],axis=0)
    K_inv = K_inv[np.newaxis,:,:].repeat(R_matrix.shape[0],axis=0)
    r_move = K@R_matrix
    r_move = r_move@K_inv - np.eye(3)  # 116，3，3

    T_move = K@T/depth               # 116,3,1
    pix_place = np.stack([pix_x,pix_y,ones],axis=0)  # 3，200000
    pix_place = pix_place[np.newaxis, :, :]
    pix_place = np.repeat(pix_place,r_move.shape[0] , axis=0)
    pix_move = r_move@pix_place 
    pix_move = pix_move + T_move

    x_move_sheet = pix_move[:,0,:].reshape(quat.shape[0],height,width)
    y_move_sheet = pix_move[:,1,:].reshape(quat.shape[0],height,width)
 
    return x_move_sheet,y_move_sheet


if __name__ == '__main__':
# 4703.863770,4750.683105,750.655371,491.003268
# 2000 1000 6472.86 6419.42 1000 500

    fx = 4703.863770
    fy = 4750.42
    cx = 750
    cy = 491
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
    # h, w = 512, 640
    h, w = 1000, 2000
    d = 20
    N = 0
    yrange = [0, 999 ]
    xrange = [1000-400 , 1000 + 400]
    var_threshold = 5000

    camnum = 12
    fB = 32504
    min_depth_percentile = 2
    max_depth_percentile = 98
    depthmapsdir = os.path.abspath('./002/depth_maps')
    outputdir = os.path.abspath('./002/depth_img') 

#  取出深度
    print("取出深度")
    depthmaps_list = os.listdir(depthmapsdir)
    depthmaps_list=sorted(filter(lambda x:"geometric" in x,depthmaps_list),key=lambda x : int(x.split('_')[0]) )
    depth_mask = np.zeros((len(depthmaps_list),h, w), dtype=np.uint8)

    for depth_index in range(len(depthmaps_list)):
        depth_mask[depth_index]=bin2depth(depth_index, os.path.join(depthmapsdir, depthmaps_list[depth_index]), outputdir)

    print("得到了深度mask")


    image_name = './04_24_002'
    # image_directory = os.path.join('.', image_name)
    # 读取图像并存储到矩阵中
    image_matrix,imagepatch_name = read_images(image_name, image_size=(h, w, 3))

    
    # 对图像矩阵进行mask处理

    for i in range(depth_mask.shape[0]):
        image_matrix[i] = image_matrix[i] * depth_mask[i].reshape(h,w,1)

    




    file_path = os.path.join('.', image_name, 'images.txt')  # 替换为你的文件路径
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    Tx_list = []
    Ty_list = []
    Tz_list = []
    Qw_list = []
    Qx_list = []
    Qy_list = []
    Qz_list = []
    name_list =[]
    for line in lines:
        if '.jpg' in line or '.png'in line :
            data = line.strip().split()
            # 获取最后3个参数
            params = np.float64(data[-5:-2])
            Qw_list.append(np.float64(data[-9]))
            Qx_list.append(np.float64(data[-8]))
            Qy_list.append(np.float64(data[-7]))
            Qz_list.append(np.float64(data[-6]))
            Tx_list.append(params[0])
            Ty_list.append(params[1])
            Tz_list.append(params[2])

            name_list.append(data[-1].split('_')[0])

    name_list=np.array(list(map(int,name_list)))

    # 使用zip将多个列表组合成一个元组的列表  
    combined = list(zip(name_list,Tx_list, Ty_list, Tz_list,Qx_list,Qy_list,Qz_list,Qw_list))  
    
    # 使用sorted函数按name的值进行排序  
    sorted_combined = sorted(combined, key=lambda item: item[0]) 

    # 使用zip将排序后的元组列表解开成单独的列表  
    name_list,Tx_list, Ty_list, Tz_list,Qx_list,Qy_list,Qz_list,Qw_list = zip(*sorted_combined)
    Tx_list = list(Tx_list)
    Ty_list = list(Ty_list)
    Tz_list = list(Tz_list)
    Qx_list = list(Qx_list)
    Qy_list = list(Qy_list)
    Qz_list = list(Qz_list)
    Qw_list = list(Qw_list)


    image_matrix = torch.from_numpy(image_matrix)
    image_matrix = image_matrix.type(torch.float64)

    # 制作mask 
    depth_mask = torch.from_numpy(depth_mask)
    depth_mask = depth_mask.type(torch.float64)

    quat = torch.stack([torch.Tensor(Qx_list), torch.Tensor(Qy_list), torch.Tensor(Qz_list), torch.Tensor(Qw_list)], dim=1)


    # 输出矩阵的形状
    image_matrix = image_matrix.unsqueeze(dim=0)
    depth_mask = depth_mask.unsqueeze(dim=0)
    Tx_list= torch.Tensor(Tx_list)
    Ty_list= torch.Tensor(Ty_list)
    Tz_list= torch.Tensor(Tz_list)
    Tx_list= Tx_list.type(torch.float64)
    Ty_list= Ty_list.type(torch.float64)
    Tz_list= Tz_list.type(torch.float64)
    Tx_max = torch.max(Tx_list) - torch.min(Tx_list)
    Ty_max = torch.max(Ty_list) - torch.min(Ty_list)


    if Tx_max > Ty_max:
        d_min = (Tx_max * fx / 2) / (w * 1 / 2)
        d_max = (Tx_max * fx / 2) / (w * 1 / 20)
        print("x is max")
    else:
        d_min = (Ty_max * fy / 2) / (h * 1 / 2)
        d_max = (Ty_max * fy / 2) / (h * 1 / 20)
        print("y is max")
    print("d_max={}".format(d_max))

    search_range=d_max-d_min
    stop_threshold=0.1
    start_depth = d_min
    end_depth = d_max
    delta_depth = (d_max - d_min)*0.618


    save_path = f"./image_sai_out/{image_name.split('/')[-1]}_matting"
    if not os.path.exists(save_path):
        os.makedirs(f"{save_path}", exist_ok=True)

    
    for best_focus_depth in np.arange(15.30,16.0,0.05):
        print(f'best_focus_depth={best_focus_depth}')
        x_move , y_move =  pix_move(w,h,quat,best_focus_depth,K,Tx_list,Ty_list,Tz_list)
        x_move = torch.from_numpy(x_move)
        y_move = torch.from_numpy(y_move)
        x_move = x_move.unsqueeze(dim=0)
        y_move = y_move.unsqueeze(dim=0)
    
        refocus_images_R = dcn_warp(image_matrix[..., 0], x_move , y_move)
        refocus_images_G = dcn_warp(image_matrix[..., 1], x_move , y_move)
        refocus_images_B = dcn_warp(image_matrix[..., 2], x_move , y_move)
        refoucs_mask = dcn_warp(depth_mask, x_move , y_move)

        refocus_images = torch.stack([refocus_images_R,refocus_images_G,refocus_images_B],dim=-1)


        ref_image = nonzero_mean(refocus_images,refoucs_mask, dim=1, keepdim=True).squeeze()
        image = ref_image/ref_image.max()

   
        # new_image = np.zeros_like(image)
        # if image.shape[-1] == 3:
        #     new_image[:,:,0] = exposure.equalize_adapthist(image[:,:,0])
        #     new_image[:,:,1] = exposure.equalize_adapthist(image[:,:,1])
        #     new_image[:,:,2] = exposure.equalize_adapthist(image[:,:,2])
        # else:
        #     new_image = exposure.equalize_adapthist(image)
        new_image = np.uint8(np.clip(image*255., 0, 255))
        cv2.imwrite(f"{save_path}/{best_focus_depth:.3f}.png", new_image)
        print(f'OK,{image_name} is done!')
    #ref_image[ref_image > 3 * torch.mean(ref_image)] = 3 * torch.mean(ref_image)
    #imin, imax = ref_image.min(), ref_image.max()
    #ref_image = 255 * (ref_image - imin) / (imax - imin + 1e-5)
    #cv2.imwrite(f"{save_path}/pred_{best_focus_depth:.3f}.png", ref_image.numpy())
    #cv2.imwrite(f"{save_path}/cut_{best_focus_depth:.3f}.png", ref_image[yrange[0]:yrange[1], xrange[0]:xrange[1]].numpy())
    