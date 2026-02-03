from plyfile import PlyData

folder = "globe_embedding_32"

initial_points = PlyData.read("output/" + folder + "/input.ply")
gaussians = PlyData.read("output/" + folder + "/point_cloud/iteration_120000/point_cloud.ply")

print("Folder: ", folder)
print("Initial Points: ", len(initial_points['vertex'].data))
print("Number of Gaussians: ", len(gaussians['vertex'].data))