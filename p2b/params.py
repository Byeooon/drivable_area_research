import numpy as np

Rt_LtoC = np.array([
            [0., -1., 0., 0.],
            [0., 0., -1., 0.],
            [1., 0., 0., 0.]
            ])

Rt_2to3 = np.array([
            [0., -1., 0., -1.5],
            [0., 0., -1., 0.],
            [1., 0., 0., 0.]
            ])

R0 = np.array([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]
            ])

P = np.array([
        [514.1332824271614, 0., 640.0, 0.],
        [0., 514.1332824271612, 360.0, 0.],
        [0., 0., 1., 0.]
        ])

cam2tocam3 = np.array([
                [1., 0., 0., -1.5],
                [0., 1., 0., 0.],
                [0., 0., 1., 0.]
                ])
cam3tocam2 = np.array([
                [1., 0., 0., 1.5],
                [0., 1., 0., 0.],
                [0., 0., 1., 0.]
                ])

Params = {
    'Rt': Rt_LtoC,
    'Rt_2to3': Rt_2to3,
    'R0': R0,
    'P': P,
    
    't_x' : Rt_LtoC[0][3],
    't_y' : Rt_LtoC[1][3],
    't_z' : Rt_LtoC[2][3],
    
    'b_x' : P[0][3],
    'b_y' : P[1][3],
    'b_z' : P[2][3],
}