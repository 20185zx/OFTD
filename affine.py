import torch
import torch.nn as nn
import torch.nn.functional as F
def get_Dt_kernel(x, y):
    kernel_conv = torch.zeros((x.shape[0], 5, 5), device=0)
    kernel_conv[:, 0, 0] = torch.mul(torch.pow(x, 4) / 24 - torch.pow(x, 3) / 12 - torch.pow(x, 2) / 24 + x / 12,
                                     torch.pow(y, 4) / 24 + torch.pow(y, 3) / 12 - torch.pow(y, 2) / 24 - y / 12)
    kernel_conv[:, 0, 1] = torch.mul(-torch.pow(x, 4) / 6 + torch.pow(x, 3) / 6 + torch.pow(x, 2) * 2 / 3 - x * 2 / 3,
                                     torch.pow(y, 4) / 24 + torch.pow(y, 3) / 12 - torch.pow(y, 2) / 24 - y / 12)
    kernel_conv[:, 0, 2] = torch.mul(torch.pow(x, 4) / 4 - torch.pow(x, 2) * 5 / 4 + 1,
                                     torch.pow(y, 4) / 24 + torch.pow(y, 3) / 12 - torch.pow(y, 2) / 24 - y / 12)
    kernel_conv[:, 0, 3] = torch.mul(-torch.pow(x, 4) / 6 - torch.pow(x, 3) / 6 + torch.pow(x, 2) * 2 / 3 + x * 2 / 3,
                                     torch.pow(y, 4) / 24 + torch.pow(y, 3) / 12 - torch.pow(y, 2) / 24 - y / 12)
    kernel_conv[:, 0, 4] = torch.mul(torch.pow(x, 4) / 24 + torch.pow(x, 3) / 12 - torch.pow(x, 2) / 24 - x / 12,
                                     torch.pow(y, 4) / 24 + torch.pow(y, 3) / 12 - torch.pow(y, 2) / 24 - y / 12)

    kernel_conv[:, 1, 0] = torch.mul(torch.pow(x, 4) / 24 - torch.pow(x, 3) / 12 - torch.pow(x, 2) / 24 + x / 12,
                                     -torch.pow(y, 4) / 6 - torch.pow(y, 3) / 6 + torch.pow(y, 2) * 2 / 3 + y * 2 / 3)
    kernel_conv[:, 1, 1] = torch.mul(-torch.pow(x, 4) / 6 + torch.pow(x, 3) / 6 + torch.pow(x, 2) * 2 / 3 - x * 2 / 3,
                                     -torch.pow(y, 4) / 6 - torch.pow(y, 3) / 6 + torch.pow(y, 2) * 2 / 3 + y * 2 / 3)
    kernel_conv[:, 1, 2] = torch.mul(torch.pow(x, 4) / 4 - torch.pow(x, 2) * 5 / 4 + 1,
                                     -torch.pow(y, 4) / 6 - torch.pow(y, 3) / 6 + torch.pow(y, 2) * 2 / 3 + y * 2 / 3)
    kernel_conv[:, 1, 3] = torch.mul(-torch.pow(x, 4) / 6 - torch.pow(x, 3) / 6 + torch.pow(x, 2) * 2 / 3 + x * 2 / 3,
                                     -torch.pow(y, 4) / 6 - torch.pow(y, 3) / 6 + torch.pow(y, 2) * 2 / 3 + y * 2 / 3)
    kernel_conv[:, 1, 4] = torch.mul(torch.pow(x, 4) / 24 + torch.pow(x, 3) / 12 - torch.pow(x, 2) / 24 - x / 12,
                                     -torch.pow(y, 4) / 6 - torch.pow(y, 3) / 6 + torch.pow(y, 2) * 2 / 3 + y * 2 / 3)

    kernel_conv[:, 2, 0] = torch.mul(torch.pow(x, 4) / 24 - torch.pow(x, 3) / 12 - torch.pow(x, 2) / 24 + x / 12,
                                     torch.pow(y, 4) / 4 - torch.pow(y, 2) * 5 / 4 + 1)
    kernel_conv[:, 2, 1] = torch.mul(-torch.pow(x, 4) / 6 + torch.pow(x, 3) / 6 + torch.pow(x, 2) * 2 / 3 - x * 2 / 3,
                                     torch.pow(y, 4) / 4 - torch.pow(y, 2) * 5 / 4 + 1)
    kernel_conv[:, 2, 2] = torch.mul(torch.pow(x, 4) / 4 - torch.pow(x, 2) * 5 / 4 + 1,
                                     torch.pow(y, 4) / 4 - torch.pow(y, 2) * 5 / 4 + 1)
    kernel_conv[:, 2, 3] = torch.mul(-torch.pow(x, 4) / 6 - torch.pow(x, 3) / 6 + torch.pow(x, 2) * 2 / 3 + x * 2 / 3,
                                     torch.pow(y, 4) / 4 - torch.pow(y, 2) * 5 / 4 + 1)
    kernel_conv[:, 2, 4] = torch.mul(torch.pow(x, 4) / 24 + torch.pow(x, 3) / 12 - torch.pow(x, 2) / 24 - x / 12,
                                     torch.pow(y, 4) / 4 - torch.pow(y, 2) * 5 / 4 + 1)

    kernel_conv[:, 3, 0] = torch.mul(torch.pow(x, 4) / 24 - torch.pow(x, 3) / 12 - torch.pow(x, 2) / 24 + x / 12,
                                     -torch.pow(y, 4) / 6 + torch.pow(y, 3) / 6 + torch.pow(y, 2) * 2 / 3 - y * 2 / 3)
    kernel_conv[:, 3, 1] = torch.mul(-torch.pow(x, 4) / 6 + torch.pow(x, 3) / 6 + torch.pow(x, 2) * 2 / 3 - x * 2 / 3,
                                     -torch.pow(y, 4) / 6 + torch.pow(y, 3) / 6 + torch.pow(y, 2) * 2 / 3 - y * 2 / 3)
    kernel_conv[:, 3, 2] = torch.mul(torch.pow(x, 4) / 4 - torch.pow(x, 2) * 5 / 4 + 1,
                                     -torch.pow(y, 4) / 6 + torch.pow(y, 3) / 6 + torch.pow(y, 2) * 2 / 3 - y * 2 / 3)
    kernel_conv[:, 3, 3] = torch.mul(-torch.pow(x, 4) / 6 - torch.pow(x, 3) / 6 + torch.pow(x, 2) * 2 / 3 + x * 2 / 3,
                                     -torch.pow(y, 4) / 6 + torch.pow(y, 3) / 6 + torch.pow(y, 2) * 2 / 3 - y * 2 / 3)
    kernel_conv[:, 3, 4] = torch.mul(torch.pow(x, 4) / 24 + torch.pow(x, 3) / 12 - torch.pow(x, 2) / 24 - x / 12,
                                     -torch.pow(y, 4) / 6 + torch.pow(y, 3) / 6 + torch.pow(y, 2) * 2 / 3 - y * 2 / 3)

    kernel_conv[:, 4, 0] = torch.mul(torch.pow(x, 4) / 24 - torch.pow(x, 3) / 12 - torch.pow(x, 2) / 24 + x / 12,
                                     torch.pow(y, 4) / 24 - torch.pow(y, 3) / 12 - torch.pow(y, 2) / 24 + y / 12)
    kernel_conv[:, 4, 1] = torch.mul(-torch.pow(x, 4) / 6 + torch.pow(x, 3) / 6 + torch.pow(x, 2) * 2 / 3 - x * 2 / 3,
                                     torch.pow(y, 4) / 24 - torch.pow(y, 3) / 12 - torch.pow(y, 2) / 24 + y / 12)
    kernel_conv[:, 4, 2] = torch.mul(torch.pow(x, 4) / 4 - torch.pow(x, 2) * 5 / 4 + 1,
                                     torch.pow(y, 4) / 24 - torch.pow(y, 3) / 12 - torch.pow(y, 2) / 24 + y / 12)
    kernel_conv[:, 4, 3] = torch.mul(-torch.pow(x, 4) / 6 - torch.pow(x, 3) / 6 + torch.pow(x, 2) * 2 / 3 + x * 2 / 3,
                                     torch.pow(y, 4) / 24 - torch.pow(y, 3) / 12 - torch.pow(y, 2) / 24 + y / 12)
    kernel_conv[:, 4, 4] = torch.mul(torch.pow(x, 4) / 24 + torch.pow(x, 3) / 12 - torch.pow(x, 2) / 24 - x / 12,
                                     torch.pow(y, 4) / 24 - torch.pow(y, 3) / 12 - torch.pow(y, 2) / 24 + y / 12)
    return kernel_conv

# def get_kernel_conv(theta):
#     kernel_conv = torch.zeros((theta.shape[0], 3, 3), device=0)
#     kernel_conv[:, 0, 0] = 0.5 * (torch.pow(torch.sin(theta), 2) + torch.sin(theta)) * torch.cos(theta).clone()
#     kernel_conv[:, 0, 2] = 0.5 * (torch.pow(torch.sin(theta), 2) - torch.sin(theta)) * torch.cos(theta).clone()
#     kernel_conv[:, 1, 0] = 0.5 * (torch.pow(torch.sin(theta), 2) + torch.sin(theta)) * (
#                 1 - torch.cos(theta)).clone()
#     kernel_conv[:, 1, 2] = 0.5 * (torch.pow(torch.sin(theta), 2) - torch.sin(theta)) * (
#                 1 - torch.cos(theta)).clone()
#     kernel_conv[:, 0, 1] = torch.pow(torch.cos(theta), 3).clone()
#     kernel_conv[:, 1, 1] = (torch.pow(torch.cos(theta), 2) * (1 - torch.cos(theta)) - 1).clone()
#     return kernel_conv
def get_kernel_conv(theta):
    temp_max = torch.max(
        torch.cat((torch.sin(theta).unsqueeze(0), torch.zeros(theta.shape, device=0).unsqueeze(0).detach()), dim=0),
        dim=0)[0]
    temp_min = -torch.min(
        torch.cat((torch.sin(theta).unsqueeze(0), torch.zeros(theta.shape, device=0).unsqueeze(0).detach()), dim=0),
        dim=0)[0]
    kernel_conv = torch.zeros((theta.shape[0], 3, 3), device=0)
    kernel_conv[:, 0, 0] = temp_min * torch.cos(theta).clone()
    kernel_conv[:, 0, 2] = temp_max * torch.cos(theta).clone()
    kernel_conv[:, 1, 0] = temp_min * (
                1 - torch.cos(theta)).clone()
    kernel_conv[:, 1, 2] = temp_max * (
                1 - torch.cos(theta)).clone()
    kernel_conv[:, 0, 1] = (1-torch.abs(torch.sin(theta))) * torch.cos(theta).clone()
    kernel_conv[:, 1, 1] = ((1-torch.abs(torch.sin(theta)))  * (1 - torch.cos(theta)) - 1).clone()
    return kernel_conv
# def get_kernel_filter(theta):
#     kernel_filter = torch.zeros((theta.shape[0], 3, 3), device=0)
#     kernel_filter[:, 2, 0] = 0.5 * (torch.pow(torch.sin(theta), 2) - torch.sin(theta)) * torch.cos(theta).clone()
#     kernel_filter[:, 2, 2] = 0.5 * (torch.pow(torch.sin(theta), 2) + torch.sin(theta)) * torch.cos(theta).clone()
#     kernel_filter[:, 1, 0] = 0.5 * (torch.pow(torch.sin(theta), 2) - torch.sin(theta)) * (
#                 1 - torch.cos(theta)).clone()
#     kernel_filter[:, 1, 2] = 0.5 * (torch.pow(torch.sin(theta), 2) + torch.sin(theta)) * (
#                 1 - torch.cos(theta)).clone()
#     kernel_filter[:, 2, 1] = torch.pow(torch.cos(theta), 3).clone()
#     kernel_filter[:, 1, 1] = (torch.pow(torch.cos(theta), 2) * (1 - torch.cos(theta)) - 1).clone()
#     return kernel_filter
def get_kernel_filter(theta):
    temp_max = torch.max(
        torch.cat((torch.sin(theta).unsqueeze(0), torch.zeros(theta.shape, device=0).unsqueeze(0).detach()), dim=0),
        dim=0)[0]
    temp_min = -torch.min(
        torch.cat((torch.sin(theta).unsqueeze(0), torch.zeros(theta.shape, device=0).unsqueeze(0).detach()), dim=0),
        dim=0)[0]
    kernel_filter = torch.zeros((theta.shape[0], 3, 3), device=0)
    kernel_filter[:, 2, 2] = temp_min * torch.cos(theta).clone()
    kernel_filter[:, 2, 0] = temp_max * torch.cos(theta).clone()
    kernel_filter[:, 1, 2] = temp_min * (
                1 - torch.cos(theta)).clone()
    kernel_filter[:, 1, 0] = temp_max * (
                1 - torch.cos(theta)).clone()
    kernel_filter[:, 2, 1] = (1-torch.abs(torch.sin(theta))) * torch.cos(theta).clone()
    kernel_filter[:, 1, 1] = ((1-torch.abs(torch.sin(theta)))  * (1 - torch.cos(theta)) - 1).clone()
    return kernel_filter
def Tilt_operator(x, theta, Bool):
    if Bool:
        kernel = get_kernel_conv(theta)
        kernel = torch.cuda.FloatTensor(kernel).unsqueeze(0).clone()
    else:
        kernel = get_kernel_filter(theta)
        kernel = torch.cuda.FloatTensor(kernel).unsqueeze(0).clone()
    x1 = torch.nn.functional.pad(x, pad=(1, 1, 1, 1), mode="circular")
    temp = x.clone()
    for i in range(x.shape[1]):
        temp[:, i, :, :] = torch.nn.functional.conv2d(x1[:, i, :, :].unsqueeze(1),
                                                        kernel[:, i, :, :].unsqueeze(1), padding=0).detach()
    return temp.detach()

def get_II(mv, kernel, shape):
    kernel = torch.cuda.FloatTensor(kernel).unsqueeze(0).clone()
    temp_II = torch.zeros(shape, device=0).detach()
    temp_II[:, :, :2, :2] = kernel[:, :, 1:3, 1:3].clone()
    temp_II[:, :, :2, -1] = kernel[:, :, 1:3, 0].clone()
    temp_II[:, :, -1, :2] = kernel[:, :, 0, 1:3].clone()
    temp_II[:, :, -1, -1] = kernel[:, :, 0, 0].clone()

    ans = torch.fft.fft2(temp_II).cuda()

    ans = (mv * torch.pow(torch.abs(ans),2) + 1 + mv).clone()
#     temp_II = torch.cat((temp_II.unsqueeze(4), torch.zeros(temp_II.unsqueeze(4).shape, device=0)),
#                             dim=4).cuda().detach().clone()
#     ans = torch.fft(temp_II, 2).cuda().detach().clone()

#     ans = mv * (torch.pow(ans[:, :, :, :, 0], 2) + torch.pow(ans[:, :, :, :, 1], 2)) + 1 + mv

    return ans.detach()

def get_rotate_matrix(theta):
    rotate_matrix = torch.zeros((theta.shape[0], 2, 3), device=0)
    rotate_matrix[:, 0, 0] = torch.cos(theta)
    rotate_matrix[:, 0, 1] = torch.sin(-theta)
    rotate_matrix[:, 1, 0] = torch.sin(theta)
    rotate_matrix[:, 1, 1] = torch.cos(theta)
    return rotate_matrix

def get_Scale_matrix(Scale_factor):
    Scale_matrix = torch.zeros((Scale_factor.shape[0], 2, 3), device=0)
    Scale_matrix[:, 0, 0] = Scale_factor
    Scale_matrix[:, 1, 1] = Scale_factor
    return Scale_matrix

def get_move_matrix(x,y):
    move_matrix = torch.zeros((x.shape[0], 2, 3), device=0)
    move_matrix[:, 0, 0] = torch.ones((x.shape[0]),device=0)
    move_matrix[:, 1, 1] = torch.ones((x.shape[0]),device=0)
    move_matrix[:, 0, 2] = -x
    move_matrix[:, 1, 2] = -y
    return move_matrix

def get_affine_matrix(x,y,Scale_factor,theta):
    move_matrix = torch.zeros((x.shape[0], 2, 3), device=0)
    move_matrix[:, 0, 0] = Scale_factor*torch.cos(theta)
    move_matrix[:,1, 1] = Scale_factor*torch.cos(theta)
    move_matrix[:, 0, 1] = -Scale_factor*torch.sin(theta)
    move_matrix[:,1, 0] = Scale_factor*torch.sin(theta)
    move_matrix[:, 0, 2] = torch.cos(theta)*x-torch.sin(theta)*y
    move_matrix[:,1, 2] = torch.sin(theta)*x+torch.cos(theta)*y
    return move_matrix

def get_affine_matrix_1(x,y,Scale_factor,theta):
    move_matrix = torch.zeros((x.shape[0],x.shape[1], 2, 3), device=0)
    move_matrix[:,:, 0, 0] = Scale_factor*torch.cos(theta)
    move_matrix[:,:, 1, 1] = Scale_factor*torch.cos(theta)
    move_matrix[:,:, 0, 1] = -Scale_factor*torch.sin(theta)
    move_matrix[:,:, 1, 0] = Scale_factor*torch.sin(theta)
    move_matrix[:,:, 0, 2] = torch.cos(theta)*x-torch.sin(theta)*y
    move_matrix[:,:, 1, 2] = torch.sin(theta)*x+torch.cos(theta)*y
    return move_matrix

def fcn(num_input_channels=200, num_output_channels=128, num_hidden=[300, 500, 800, 1000, 800, 500, 300]):


    model = nn.Sequential()
    model.add(nn.Linear(num_input_channels, num_hidden[0], bias=True))
    model.add(nn.LeakyReLU(0.05))

    for i in range(len(num_hidden)-1):
        model.add(nn.Linear(num_hidden[i], num_hidden[i+1], bias=True))
        model.add(nn.LeakyReLU(0.05))

    model.add(nn.Linear(num_hidden[len(num_hidden)-1], num_output_channels))

    return model

def affine_B(B, x,y,rotate_theta,Scale_factor,pad_num):
    h = B.shape[2]
    w = B.shape[3]
 
    ReplicationPad = nn.ReplicationPad2d(padding=(0, 0, int((w - h) / 2), int((w - h) / 2)))
    B_pad = ReplicationPad(B)

    # 1, 32, 384, 384
    # # [3, 32, 768, 768]
    matrix = get_affine_matrix(x,y,Scale_factor,rotate_theta)
    print(matrix.shape)
    # 32,2,3
    affine_B = F.grid_sample(B_pad.permute(1,0,2,3), 
                             F.affine_grid(matrix,  B_pad.permute(1,0,2,3).size()))
    #print(F.affine_grid(matrix, B_pad.permute(1,0,2,3).size()).shape)
    print('aaaaaaa',affine_B.shape)
    affine_B = affine_B[:,:, pad_num + int((w - h) / 2):
                        -pad_num - int((w - h) / 2), pad_num: -pad_num]
    #affine_B = affine_B[:,:, pad_num: -pad_num,pad_num + int((h - w) / 2):
                        #-pad_num - int((h - w) / 2)]
    print('bbb',affine_B.shape)
    return affine_B.permute(1,0,2,3)


def affine_B_1(B, x,y,rotate_theta,Scale_factor,pad_num):
    h = B.shape[2]
    w = B.shape[3]
    ReplicationPad = nn.ReplicationPad2d(padding=(0, 0, int((w - h) / 2), int((w - h) / 2)))
    B_pad = ReplicationPad(B)
    #print(B_pad.shape)
    # 1, 32, 384, 384
    # 32,3,384,384
    matrix = get_affine_matrix(x,y,Scale_factor,rotate_theta)
    print(matrix.shape)
    # 32,2,3
    affine_B = F.grid_sample(B_pad.permute(1,0,2,3), 
                             F.affine_grid(matrix,  B_pad.permute(1,0,2,3).size()))
    #print(F.affine_grid(matrix, B_pad.permute(1,0,2,3).size()).shape)
    affine_B = affine_B[:,:, pad_num + int((w - h) / 2):
                        -pad_num - int((w - h) / 2), pad_num: -pad_num]
    #affine_B = affine_B[:,:, pad_num: -pad_num,pad_num + int((h - w) / 2):
                        #-pad_num - int((h - w) / 2)]
    return affine_B.permute(1,0,2,3)

def affine_B1(B, x, y, rotate_theta, Scale_factor, pad_num):
    h = B.shape[2]
    w = B.shape[3]
    # [1, 96, 216, 264]
    # print('bb',B.shape)
    if h > w:
        diff = h - w
        left = diff // 2
        right = diff - left
        ReplicationPad = nn.ReplicationPad2d((left, right, 0, 0))
    elif w > h:
        diff = w - h
        top = diff // 2
        bottom = diff - top
        ReplicationPad = nn.ReplicationPad2d((0, 0, top, bottom))
    
    B_pad = ReplicationPad(B)
    # # 1, 96, 264, 264
    # print('pad',B_pad.shape)
    # 获取仿射矩阵
    # print(x.shape)
    matrix = get_affine_matrix(x, y, Scale_factor, rotate_theta)
    # print(matrix.shape)
    # 进行仿射变换
    affine_B = F.grid_sample(B_pad.permute(1, 0, 2, 3), 
                             F.affine_grid(matrix, B_pad.permute(1, 0, 2, 3).size()))
    
    # 根据高度和宽度的关系进行裁剪
    if h > w:
        diff = h - w
        left_crop = pad_num + diff // 2
        right_crop = pad_num + diff - diff // 2
        affine_B = affine_B[:, :, pad_num:-pad_num, left_crop : -right_crop]
    else:
        diff = w - h
        top_crop = pad_num + diff // 2
        bottom_crop = pad_num + diff - diff // 2
        affine_B = affine_B[:, :, top_crop : -bottom_crop,  pad_num:-pad_num]

    # print('aii',affine_B.shape)
    return affine_B.permute(1,0,2,3)
