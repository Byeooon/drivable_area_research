import numpy as np
import pandas as pd

heightmap_resolution = 0.02

# generate some random 3D points
points =  np.array([[x,y,z] for x in np.random.uniform(0,2,100) for y in np.random.uniform(0,2,100) for z in np.random.uniform(0,2,100)])
points_df = pd.DataFrame(points, columns = ['x','y','z'])
#didn't know if you wanted to keep the x and y columns so I made new ones.
points_df['x_normalized'] = (points_df['x']/heightmap_resolution).astype(int)
points_df['y_normalized'] = (points_df['y']/heightmap_resolution).astype(int)
tmp = points_df.groupby(['x_normalized','y_normalized'])['z'].max()

print(tmp)