'''
Vertex finding using list of points
'''
import torch
import sparseconvnet as scn
import numpy as np
from SCN import DeepVtx


def Test(weights, x, y, z, q, dtype):
    '''
    IO test
    '''
    print("python: Test")
    print("weights: ", weights)
    x = np.frombuffer(x, dtype=dtype)
    y = np.frombuffer(y, dtype=dtype)
    z = np.frombuffer(z, dtype=dtype)
    q = np.frombuffer(q, dtype=dtype)
    coords = np.stack((x, y, z), axis=1)
    ft = np.expand_dims(q, axis=1)
    print("coords: ", coords)
    print(" ft: ", ft)

    vx = np.mean(x)
    vy = np.mean(y)
    vz = np.mean(z)

    vtx = np.array([vx, vy, vz])
    return vtx.tobytes()

def SCN_Vertex(weights, x, y, z, q, dtype):
    print("python: SCN_Vertex")
    print("weights: ", weights)
    x = np.frombuffer(x, dtype=dtype)
    y = np.frombuffer(y, dtype=dtype)
    z = np.frombuffer(z, dtype=dtype)
    q = np.frombuffer(q, dtype=dtype)
    coords_np = np.stack((x, y, z), axis=1)
    ft_np = np.expand_dims(q, axis=1)
    print("coords: ", coords_np)
    print(" ft: ", ft_np)

    torch.set_num_threads(1)
    device = 'cpu'

    coords = torch.LongTensor(coords_np)
    ft = torch.FloatTensor(ft_np).to(device)
    print("coords: ", coords)
    print(" ft: ", ft)

    nIn = 1
    model = DeepVtx(dimension=3, nIn=nIn, device=device)
    model.train()
    trained_dict = torch.load(weights)

    # torch 1.0.0 seems to have 3 dims for some tensors while 1.3.1 have 4 for them
    # in that case dim=1 is an unsqueezed dim with size only 1
    for param_tensor in trained_dict:
        # print("current: ", param_tensor, "\t", model.state_dict()[param_tensor].shape)
        # print("trained: ", param_tensor, "\t", trained_dict[param_tensor].shape)
        if trained_dict[param_tensor].shape !=  model.state_dict()[param_tensor].shape:
            trained_dict[param_tensor] = torch.squeeze(trained_dict[param_tensor], dim=1)
        # print("squeezed: ", param_tensor, "\t", trained_dict[param_tensor].shape)
    model.load_state_dict(trained_dict)

    prediction = model([coords,ft])
    print(prediction)
    pred_np = prediction.cpu().detach().numpy()
    pred_np = pred_np[:,1] - pred_np[:,0]
    print('pred_np', pred_np)
    
    dnn_pred_idx = np.argmax(pred_np)
    coords_p_dnn = coords_np[dnn_pred_idx]
    print('coords_p_dnn', coords_p_dnn)

    return coords_p_dnn.tobytes()

if __name__ == '__main__':
    dtype = 'f'
    weights = '/lbne/u/hyu/lbne/uboone/t48k-m16-l5-lr5d-res0.5-CP24.pth'
    x = np.array([00.0, 01.0, 02.0], dtype=dtype).tobytes()
    y = np.array([10.0, 11.0, 12.0], dtype=dtype).tobytes()
    z = np.array([20.0, 21.0, 22.0], dtype=dtype).tobytes()
    q = np.array([1.0, 2.0, 3.0], dtype=dtype).tobytes()

    SCN_Vertex(weights, x, y, z, q, dtype)