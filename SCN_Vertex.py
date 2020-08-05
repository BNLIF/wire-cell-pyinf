'''
Vertex finding using list of points
'''
import torch
import sparseconvnet as scn
import numpy as np
from SCN import DeepVtx


def Test(weights, x, y, z, q, dtype):
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
    # model.load_state_dict(torch.load(weights))
    model.load_state_dict(model.state_dict())
    print(model)

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
    x = np.array([1.1, 2.1, 3.1], dtype=dtype).tobytes()
    y = np.array([1.1, 2.1, 3.1], dtype=dtype).tobytes()
    z = np.array([1.1, 2.1, 3.1], dtype=dtype).tobytes()
    q = np.array([1.0, 1.0, 1.0], dtype=dtype).tobytes()

    SCN_Vertex(weights, x, y, z, q, dtype)