import open3d as o3d
import numpy as np


def load_mesh_from_txt(filename):
    vertices = []
    triangles = []

    with open(filename, 'r') as file:
        lines = file.readlines()

        # 第一行是顶点数量
        n = int(lines[0].strip())

        # 读取顶点坐标
        for i in range(1, n + 1):
            x, y, z = map(float, lines[i].strip().split())
            vertices.append([x, y, z])

        # 读取三角形
        for line in lines[n + 1:]:
            i, j, k = map(int, line.strip().split())
            triangles.append([i, j, k])

    return np.array(vertices), np.array(triangles)

# 读取文件
file_path = "V.txt"
data = np.loadtxt(file_path)

# 分割数据
points = data[:, :3]
vectors = data[:, 3:]
norms = np.linalg.norm(vectors, axis=1, keepdims=True)
vectors = vectors / max(norms) / 10

# 创建点云
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)


coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=1.0,  # 坐标系大小
    origin=[0, 0, 0]  # 坐标系的原点
)

# 创建LineSet来表示向量场
lines = []
colors = []

# 创建向量场箭头
for i in range(len(points)):
    lines.append([i, len(points) + i])
    colors.append([1, 0, 0])  # 使用红色表示箭头

# 创建新的点用于表示箭头的终点
end_points = points + vectors
all_points = np.vstack((points, end_points))

# 创建LineSet
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(all_points)
line_set.lines = o3d.utility.Vector2iVector(lines)
line_set.colors = o3d.utility.Vector3dVector(colors)

# 可视化点云和向量场
vis = o3d.visualization.Visualizer()
vis.create_window()

# 添加点云和向量场到可视化窗口
vis.add_geometry(pcd)
vis.add_geometry(line_set)
#vis.add_geometry(coordinate_frame)


# 读取三角网格数据
filename = 'triangles.txt'  # 替换为你的文件路径
vertices, triangles = load_mesh_from_txt(filename)

# 创建 Open3D 的 TriangleMesh 对象
mesh = o3d.geometry.TriangleMesh()
mesh.compute_vertex_normals()
mesh.vertex_normals = o3d.utility.Vector3dVector(-1 * np.asarray(mesh.vertex_normals))
mesh.vertices = o3d.utility.Vector3dVector(vertices)
mesh.triangles = o3d.utility.Vector3iVector(triangles)
line_set = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)

# 可视化三角网格
vis.add_geometry(mesh)
vis.add_geometry(line_set)

# 禁用背面剔除
opt = vis.get_render_option()
opt.mesh_show_back_face = True
# 开始可视化
vis.run()
vis.destroy_window()
