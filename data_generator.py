import open3d as o3d
# import numpy as jnp
import jax.numpy as jnp
from sklearn.neighbors import KDTree
import pickle

def place_balls(pt, normals, balls):
    # place balls onto the geometry defined by the input pt.

    # place balls
    for ball in balls:
        kd = KDTree(pt)
        inds = kd.query_radius(ball[0:3].reshape(1, -1), ball[3])[0]
        rb = jnp.linalg.norm(ball[0:3])
        cosda = jnp.dot(pt[inds], ball[0:3])/rb/jnp.linalg.norm(pt[inds], axis=1)
        R = rb*cosda+jnp.sqrt(rb**2*cosda**2-(rb**2-ball[3]**2))
        pt = pt.at[inds].set(R.reshape(-1, 1)*normals[inds])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pt)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(10))
    pcd.orient_normals_towards_camera_location()

    return pt, jnp.asarray(pcd.normals)


def indent_balls(pt, normals, balls):
    # indent balls against the geometry defined by the inout pt.

    # indent balls
    for ball in balls:
        kd = KDTree(pt)
        inds = kd.query_radius(ball[0:3].reshape(1, -1), ball[3])[0]
        rb = jnp.linalg.norm(ball[0:3])
        cosda = jnp.dot(pt[inds], ball[0:3])/rb/jnp.linalg.norm(pt[inds], axis=1)
        R = rb*cosda-jnp.sqrt(rb**2*cosda**2-(rb**2-ball[3]**2))
        pt = pt.at[inds].set(R.reshape(-1, 1)*normals[inds])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pt)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(10))
    pcd.orient_normals_towards_camera_location()

    return pt, jnp.asarray(pcd.normals)
    # todo generate test data for curved surface reoncstruction
#
# depth = 2.5
# num_points = 10000
#
# mesh = o3d.io.read_triangle_mesh("sensing_surface_distal.stl")
# pcd = mesh.sample_points_poisson_disk(num_points)
#
# pc = jnp.asarray(pcd.points)
# norms = jnp.asarray(pcd.normals)
# ind = jnp.random.randint(num_points)
# pt = pc[ind]
# kd = KDTree(pc)
# inds = kd.query_radius(pt.reshape(1, -1), depth)[0]
# dist = jnp.linalg.norm(pc[inds] - pt, axis=1)
# pc[inds] -= (depth - dist.reshape(-1, 1)) * norms[inds]
# pcd2 = o3d.geometry.PointCloud()
# pcd2.points = o3d.utility.Vector3dVector(pc)
# pcd2.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(10))
# pcd2.orient_normals_towards_camera_location()
# #o3d.visualization.draw_geometries([pcd2, o3d.geometry.TriangleMesh.create_coordinate_frame(1)])
#
# pickle.dump(jnp.asarray(pcd2.normals), open("normals_n.pkl", "wb"))
# pickle.dump(jnp.asarray(pcd.points), open("points_n.pkl", "wb"))