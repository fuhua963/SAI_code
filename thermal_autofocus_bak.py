# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
import os
import torch
from torch import Tensor
import torchvision
import matplotlib.pyplot as plt




def dcn_warp(voxelgrid: Tensor, flow_x: Tensor, flow_y: Tensor):
    # voxelgrid: [bs,ts,H,W] | flow: [bs,ts,H,W]
    bs, ts, H, W = voxelgrid.shape
    flow = torch.stack([flow_y, flow_x], dim=2)  # [bs,ts,2,H,W]
    flow = flow.reshape(bs, ts * 2, H, W)  # [bs,ts*2,H,W]
    #  单位矩阵 保证了 只对一张图处理
    weight = torch.eye(ts, device=flow.device).double().reshape(ts, ts, 1, 1)  # 返回 ts 张图 对ts张图做处理
    # 单位卷积核
    return torchvision.ops.deform_conv2d(voxelgrid, flow, weight)  # [bs,ts,H,W]


def nonzero_mean(tensor: Tensor, dim: int, keepdim: bool = False, eps=1e-3):
    numel = torch.sum((tensor > eps).float(), dim=dim, keepdim=keepdim)
    value = torch.sum(tensor, dim=dim, keepdim=keepdim)
    return value / (numel + eps)


def read_images(directory_path, image_size):
    # 获取目录中所有图像文件的文件名
    image_files = [f for f in os.listdir(directory_path) if f.endswith('.jpg') or f.endswith('.png')]
    global N
    N = len(image_files)
    # 初始化矩阵
    images_matrix = np.zeros((len(image_files), *image_size), dtype=np.float64)
    # 读取每张图像并存储到矩阵中
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(directory_path, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # image = image.astype(np.float64) / 255.0  # [1,N,h,w]
        images_matrix[i] = np.array(image)
    return images_matrix


def apply_canny(image):

    image=image.reshape((h,w))
    return cv2.Canny(image, 100, 200)

def apply_var(image_pixel):
    image_pixel = image_pixel[image_pixel>1e-3]
    var_image = np.var(image_pixel)
    return var_image


if __name__ == '__main__':
    # opencv 标定
    K = np.array([[796.17394082, 0, 317.5847932],
                  [0, 791.09602179, 222.86617395],
                  [0, 0, 1]])
    # 768.000000, 768.000000, 320.000000, 256.00000
    fx = 768
    fy = 768
    h, w = 512, 640
    # h, w = 1080, 1920
    d = 20
    N = 0
    yrange = [256 - 100, 256 + 100]
    xrange = [320 - 150, 320 + 150]
    # yrange = [260 - 50, 260 + 50]
    # xrange = [500 - 60, 500 + 60]

    image_name = '25m_thermal'
    image_directory = os.path.join('.', image_name)
    # 读取图像并存储到矩阵中
    image_matrix = read_images(image_directory, image_size=(h, w))
    image_matrix = torch.from_numpy(image_matrix)
    # 输出矩阵的形状
    image_matrix = image_matrix.unsqueeze(dim=0)
    print("图像矩阵的形状:", image_matrix.shape)
    # 读取文件  位姿

    file_path = os.path.join('.', image_name, 'images.txt')  # 替换为你的文件路径
    with open(file_path, 'r') as file:
        lines = file.readlines()
    Tx_list = np.zeros(N)
    Ty_list = np.zeros(N)
    num = 0
    for line in lines:
        if '.jpg' in line:
            data = line.strip().split()
            # 获取最后四个参数
            params = np.float64(data[-5:-2])
            Tx_list[num] = params[0]
            Ty_list[num] = params[1]
            num += 1
    del num
    Tx_list = torch.from_numpy(Tx_list)
    Ty_list = torch.from_numpy(Ty_list)
    Tx_max = torch.max(Tx_list) - torch.min(Tx_list)
    Ty_max = torch.max(Ty_list) - torch.min(Ty_list)

    if Tx_max > Ty_max:
        d_min = (Tx_max * fx / 2) / (w * 1 / 2)
        d_max = (Tx_max * fx / 2) / (w * 1 / 7)
        print("x is max")
    else:
        d_min = (Ty_max * fy / 2) / (h * 1 / 2)
        d_max = (Ty_max * fy / 2) / (h * 1 / 7)
        print("y is max")
    print("d_max={}".format(d_max))
    # exit()
    # d_for_img_list = np.linspace(d_min, d_max, num=60)
    Tx_list = Tx_list.unsqueeze(dim=0)
    Ty_list = Ty_list.unsqueeze(dim=0)
    Tx_list = Tx_list.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
    Ty_list = Ty_list.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)

    refocus_assess_list = []
    image_var_list=[]
    k_d_list = []
    laplace_list = []
    well_point_list=[]
    # mulview_candy_image = np.zeros((N, h, w))

    save_path = f"./{image_name}_output"
    os.makedirs(f"{save_path}", exist_ok=True)
    # 图片分块


    search_range=d_max-d_min
    stop_threshold=0.1
    start_depth = d_min
    end_depth = d_max
    delta_depth = (d_max - d_min)*0.618

    while (delta_depth > stop_threshold):
        fst_focus_d = start_depth + delta_depth / 2
        sec_focus_d = end_depth - delta_depth / 2


        fst_refocus_images = dcn_warp(image_matrix, fx * Tx_list / fst_focus_d, fy * Ty_list / fst_focus_d)
        fst_ref_image = nonzero_mean(fst_refocus_images, dim=1, keepdim=True).squeeze()
        sec_refocus_images = dcn_warp(image_matrix, fx * Tx_list / sec_focus_d, fy * Ty_list / sec_focus_d)
        sec_ref_image = nonzero_mean(sec_refocus_images, dim=1, keepdim=True).squeeze()

        fst_image_var = torch.var(fst_ref_image[yrange[0]:yrange[1], xrange[0]:xrange[1]])
        sec_image_var = torch.var(sec_ref_image[yrange[0]:yrange[1], xrange[0]:xrange[1]])

        if(fst_image_var>sec_image_var):
            start_depth = start_depth
            end_depth = start_depth + delta_depth
        else:
            start_depth = end_depth - delta_depth
            end_depth = end_depth

        delta_depth = (end_depth - start_depth) * 0.618
        # print(f'delta_depth={delta_depth}')
        print(f'fst_focus_d={fst_focus_d},sec_focus_d={sec_focus_d}')
        print(f'fst_image_var={fst_image_var},sec_focus_d={sec_image_var}\n')
    best_focus_depth = (start_depth+end_depth)/2
    print(f'best_focus_depth={best_focus_depth}')
    refocus_images = dcn_warp(image_matrix, fx * Tx_list / best_focus_depth, fy * Ty_list / best_focus_depth)
    ref_image = nonzero_mean(refocus_images, dim=1, keepdim=True).squeeze()  # [h,w]
    ref_image[ref_image > 3 * torch.mean(ref_image)] = 3 * torch.mean(ref_image)
    imin, imax = ref_image.min(), ref_image.max()
    ref_image = 255 * (ref_image - imin) / (imax - imin + 1e-5)
    cv2.imwrite(f"{save_path}/pred_{best_focus_depth:.3f}.png", ref_image.numpy())
    cv2.imwrite(f"{save_path}/cut_{best_focus_depth:.3f}.png", ref_image[yrange[0]:yrange[1], xrange[0]:xrange[1]].numpy())








    # for i, d in enumerate(d_for_img_list):
    #     # k_d=float((1.0+i/50))
    #     d_for_img = d
    #     refocus_images = dcn_warp(image_matrix, fx * Tx_list / d_for_img, fy * Ty_list / d_for_img)
    #     ref_image = nonzero_mean(refocus_images, dim=1, keepdim=True).squeeze()  # [h,w]
        # ref_image[ref_image > 3 * torch.mean(ref_image)] = 3 * torch.mean(ref_image)
        # imin, imax = ref_image.min(), ref_image.max()
    #     image_var = torch.var(ref_image[yrange[0]:yrange[1], xrange[0]:xrange[1]])
    #     image_var_list.append(image_var)
    #     k_d_list.append(d_for_img)
    #     ref_image = 255 * (ref_image - imin) / (imax - imin + 1e-5)
    #     cv2.imwrite(f"{save_path}/pred_{d_for_img:.3f}.png", ref_image.numpy())
    #     cv2.imwrite(f"{save_path}/cut_{d_for_img:.3f}.png", ref_image[yrange[0]:yrange[1], xrange[0]:xrange[1]].numpy())
    #     print(f"num={i}\n")
     
        

    #  存储list

    # k_d_list= np.array(k_d_list).astype(np.float32)
    # image_var_list=np.array(image_var_list).astype(np.float32)
    # plt.plot(k_d_list,image_var_list)
    # plt.show()
    # np.savetxt(fname=f"{image_name}_var.csv", X=(k_d_list, image_var_list), fmt="%f", delimiter=",")
