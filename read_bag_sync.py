import rosbag
import os
import numpy as np
import cv2
from cv_bridge import CvBridge
import torch
from torch import Tensor
import torchvision
import tqdm


def dcn_warp(voxelgrid: Tensor, flow_x: Tensor, flow_y: Tensor):
    # voxelgrid: [bs,ts,H,W] | flow: [bs,ts,H,W]
    bs, ts, H, W = voxelgrid.shape
    flow = torch.stack([flow_y, flow_x], dim=2)  # [bs,ts,2,H,W]
    flow = flow.reshape(bs, ts * 2, H, W)  # [bs,ts*2,H,W]
    #  单位矩阵 保证了 只对一张图处理
    weight = torch.eye(ts, device=flow.device).double().reshape(ts, ts, 1, 1)  # 返回 ts 张图 对ts张图做处理
    # 单位卷积核
    return torchvision.ops.deform_conv2d(voxelgrid, flow, weight)  # [bs,ts,H,W]


class BagReader:
    def __init__(self,base_path,occ_path,gt_path,sn,v,d):
        self.bridge = CvBridge()
        self.occ_bag_path = f'{base_path}/{occ_path}'
        self.gt_bag_path = f'{base_path}/{gt_path}'
        # intr
        self.fw,self.fh,self.froi = 2448,2048,1000
        self.pw,self.ph,self.proi = 1280,720,600
        self.fk,self.fp = np.array([[3995.8113,0.,413.5658],[0.,3897.2876,440.9243],[0.,0.,1.]]),np.array([-0.1092,0.9235,-0.0053,-0.0029,0.])
        self.pk,self.pp = np.array([[2541.2043,0.,298.1284],[0.,2539.6931,320.4003],[0.,0.,1.]]),np.array([-0.0018,-10.3420,0.,0.0010,0.0057])
        # other param
        self.v,self.d = v,d
        # output
        self.save_dir = os.path.abspath(f'./event/0625')
        os.makedirs(self.save_dir,exist_ok=True)
        self.save_path = f"{self.save_dir}/{sn:04d}.npz"
        self.sn = sn
    
    def check(self):
        if not os.path.exists(self.occ_bag_path) or not os.path.exists(self.gt_bag_path):
            print(f"No such bag path {self.occ_bag_path}.")
            return False
        if os.path.exists(self.save_path):
            print(f"Skip {self.save_path}.")
            return False
        self.occ_bag,self.gt_bag = rosbag.Bag(self.occ_bag_path),rosbag.Bag(self.gt_bag_path)
        return True
        
    def readbag(self):
        event_topic = '/prophesee/camera/cd_events_buffer'
        frame_topic = '/camera/image_raw'
        self.event_list = []
        self.gt_list,self.gt_tss_list = [],[]
        ## Event
        event_topic_counts = self.occ_bag.get_message_count(topic_filters=[event_topic])
        with tqdm.tqdm(range(event_topic_counts)) as pbar:
            for topic, msgs, t in self.occ_bag.read_messages(topics=[event_topic]):
                self.event_list += msgs.events
                pbar.update(1)
        ex = np.array(list(map(lambda event:event.x,self.event_list)))
        ey = np.array(list(map(lambda event:event.y,self.event_list)))
        et = np.array(list(map(lambda event:event.ts.secs+event.ts.nsecs*1e-9,self.event_list)))
        ep = np.array(list(map(lambda event:int(event.polarity),self.event_list)))
        roi_x0,roi_x1 = (self.pw-self.proi)//2, (self.pw-self.proi)//2+self.proi
        roi_y0,roi_y1 = (self.ph-self.proi)//2, (self.ph-self.proi)//2+self.proi
        index = (ex>=roi_x0)&(ex<=roi_x1)&(ey>=roi_y0)&(ey<roi_y1)
        ex,ey,et,ep = ex[index]-roi_x0,ey[index]-roi_y0,et[index],ep[index]
        estart = et.min()
        et -= estart
        events = np.stack([ex,ey,et,ep],axis=1)
        print(f"[Events] shape: {events.shape}")
        ## Frame
        for topic, msgs, t in self.gt_bag.read_messages():
            if topic == '/camera/image_raw':  
                image = self.bridge.imgmsg_to_cv2(msgs,"mono8")
                image = np.fliplr(image)
                self.gt_list.append(image)
                tss = msgs.header.stamp.secs+msgs.header.stamp.nsecs*1e-9
                self.gt_tss_list.append(tss)
                cv2.imshow("frame",image)
                cv2.waitKey(1)
        # check
        os.makedirs(f'{self.save_dir}/gt/{self.sn:04d}',exist_ok=True)
        for idx,each in enumerate(self.gt_list): cv2.imwrite(f'{self.save_dir}/gt/{self.sn:04d}/{idx:04d}.png',each) 
        self.gt_tss_list = np.array(self.gt_tss_list)
        t_min = self.gt_tss_list.min()
        self.gt_tss_list -= t_min
        self.gt = np.array(self.gt_list)
        roi_x0,roi_x1 = (self.fw-self.froi)//2, (self.fw-self.froi)//2+self.froi
        roi_y0,roi_y1 = (self.fh-self.froi)//2, (self.fh-self.froi)//2+self.froi
        self.gt = self.gt[:,roi_y0:roi_y1,roi_x0:roi_x1]      
        self.gt = np.array([cv2.undistort(each,self.fk,self.fp) for each in self.gt])
        #### reproject
        y1,x1 = torch.meshgrid(torch.arange(self.froi),torch.arange(self.froi),indexing="ij")
        y1,x1 = y1.double(),x1.double()
        ones = torch.ones_like(x1)
        pts1 = torch.stack([x1,y1,ones],dim=-1).reshape(-1,3).t() # [3,H*W]
        k1_inv = torch.inverse(torch.from_numpy(self.fk))
        pts_world = k1_inv @ pts1 # [3,H*W]
        pts2 = torch.from_numpy(self.pk) @ pts_world
        pts2 /= pts2[2,:].clone()
        gt = torch.from_numpy(self.gt).double().unsqueeze(0) # [1,N,H,W]
        count = torch.ones_like(gt)
        num_images = gt.shape[1]
        flow_x = (pts2[0,:]-pts1[0,:]).reshape(1,1,self.froi,self.froi).repeat(1,num_images,1,1)
        flow_y = (pts2[1,:]-pts1[1,:]).reshape(1,1,self.froi,self.froi).repeat(1,num_images,1,1)
        reproj_gt = dcn_warp(gt,flow_x,flow_y)[...,:self.proi,:self.proi].squeeze() # [N,H,W]
        reproj_count = dcn_warp(count,flow_x,flow_y)[...,:self.proi,:self.proi].squeeze() # [N,H,W]
        reproj_gt /= (reproj_count+1e-6)
        reproj_gt = reproj_gt.numpy().astype(np.uint8)
        # check 
        os.makedirs(f'{self.save_dir}/gt_reproj/{self.sn:04d}',exist_ok=True)
        for idx,each in enumerate(reproj_gt): 
            cv2.imwrite(f'{self.save_dir}/gt_reproj/{self.sn:04d}/{idx:04d}.png',each)
        print("")


        # # warp flir to prophesee
        # pix_x = np.arange(0,self.froi)
        # pix_y = np.arange(0,self.froi)
        # pix_x , pix_y = np.meshgrid(pix_x,pix_y)
        # pix_x = pix_x.reshape(-1)
        # pix_y = pix_y.reshape(-1)
        # ones = np.ones_like(pix_x)
        # pix_place = np.stack([pix_x,pix_y,ones],axis=0)  # 3，200000()
        # pix_place = pix_place[np.newaxis, :, :].repeat(len(self.gt_list),axis=0) # N,3,xxx
        # fk_inv = np.linalg.inv(self.fk)                     # 3,3
        # f2p_pix = self.pk@fk_inv
        # f2p_pix = f2p_pix[np.newaxis,...].repeat(len(self.gt_list),axis=0)    # N 3,3 
        # f2p_pix = f2p_pix@pix_place
        # pix_move = f2p_pix - pix_place                     # delta pix

        # x_move= pix_move[:,0,:].reshape(len(self.gt_list),self.froi,self.froi)
        # y_move= pix_move[:,1,:].reshape(len(self.gt_list),self.froi,self.froi)

        # x_move = torch.from_numpy(x_move)
        # y_move = torch.from_numpy(y_move)
        # x_move = x_move.unsqueeze(dim=0)
        # y_move = y_move.unsqueeze(dim=0)
        # x_move = x_move.expand(-1,-1,self.froi,self.froi)
        # y_move = y_move.expand(-1,-1,self.froi,self.froi)
        # f2p_image = dcn_warp(undistort_image, x_move , y_move)
        # f2p_image = f2p_image.squeeze()
        # # f2p_image = f2p_image[:,0:self.proi,0:self.proi]       # 裁减600X600


        # # event acc 
        # ref_t = et[int(0.5*ex.shape[0])]
        # dt = et - ref_t
        # dx = dt * v * self.pk[0,0] / d
        # event_x = ex + np.round(dx)
        # event_x = np.clip(event_x, 0, self.proi-1)

        # # Generate event_tensor
        # time_step = 64
        # minT = et.min()
        # maxT = et.max()
        # interval = (maxT - minT) / time_step
        
        # # convert events to event tensor
        # pos = np.zeros((time_step, self.proi, self.proi))
        # neg = np.zeros((time_step, self.proi, self.proi))
        # T,H,W = pos.shape
        # pos = pos.ravel()
        # neg = neg.ravel()
        # ind = (et / interval).astype(int)
        # ind[ind == T] -= 1
        # event_x = event_x.astype(int)

        # event_y = ey.astype(int)
        # pos_ind = ep == 1
        # neg_ind = ep == 0

        # np.add.at(pos, event_x[pos_ind] + event_y[pos_ind]*W + ind[pos_ind]*W*H, 1)
        # np.add.at(neg, event_x[neg_ind] + event_y[neg_ind]*W + ind[neg_ind]*W*H, 1)
        # pos = np.reshape(pos, (T,H,W))
        # neg = np.reshape(neg, (T,H,W))
        # sum_pos = np.sum(pos+neg,axis=0)
        # sum_pos[sum_pos>3*np.mean(sum_pos)]=np.mean(sum_pos)
        # sum_pos/= np.max(sum_pos)
        # sum_pos*=255
        # os.makedirs(f'{self.save_dir}/event/{self.sn:04d}',exist_ok=True)
        # cv2.imwrite(f'{self.save_dir}/event/{self.sn:04d}/pos.png',sum_pos)
        


        # # check
        # os.makedirs(f'{self.save_dir}/gt/{self.sn:04d}',exist_ok=True)
        # f2p_image = np.uint8(np.array(f2p_image))
        # for idx,gt in enumerate(f2p_image): cv2.imwrite(f'{self.save_dir}/gt/{self.sn:04d}/{idx:04d}.png',gt) 
        # for free_image in f2p_image :
        #     cv2.imshow("frame_warp",free_image)
        #     cv2.waitKey(1)


        # npz save

        # np.savez(self.save_path,v=np.array(self.v),fk=self.fk,fp=self.fp,size=np.array([self.proi,self.proi]),depth=d,events=events,
        #          occ_free_aps=np.array(f2p_image),occ_free_aps_ts=np.array(self.gt_tss_list))

        print(f"Finish writing file:{self.save_path}")

if __name__ == "__main__":
    width_s = 1000
    height_s = 1000
    offx_s = 724
    offy_s = 524
    roi_x0 = int(340)
    roi_y0 = int(60)
    roi_x1 = int(939)
    roi_y1 = int(659)
    base_path = os.path.abspath('./simulation_bag_0625')
    v,d = 0.213,1.15
    bag_list = os.listdir(base_path)
    occ_bag_list = sorted(list(filter(lambda x: x.startswith('occ'),bag_list)))
    gt_bag_list = sorted(list(filter(lambda x: x.startswith('gt'),bag_list)))
    for sn,(occ_path,gt_path) in enumerate(zip(occ_bag_list,gt_bag_list)):
        reader = BagReader(base_path,occ_path,gt_path,sn=sn,v=v,d=d)
        flag = reader.check()
        if flag: reader.readbag()