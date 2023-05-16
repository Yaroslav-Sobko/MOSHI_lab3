import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


sequence = []
N = 1000
k = 3

for i in range(N):
	temp = []
	x = random.uniform(0, 1)
	temp.append(x)
	y = random.uniform(0, 1)
	temp.append(y)
	sequence.append(temp)


def euclidean_distance(arr_center, arr):
	return np.sqrt((arr_center[0] - arr[0]) ** 2 + (arr_center[1] - arr[1]) ** 2)


# Вибір к центрів
def rndCenterGenerator():
	temp_centroids = []
	rndForCenter = random.sample(range(N), k)
	for i in rndForCenter:
		temp_centroids.append(sequence[i])

	# print(temp_centroids)
	return temp_centroids


#################################################################################

def findNewCenters(clusters, centroids):
	# Знаходження нових центрів
	for i in range(k):
		median_point = [0, 0]
		number_of_elem = 0
		for j in range(len(clusters[i])):
			median_point[0] += clusters[i][j][0]
			median_point[1] += clusters[i][j][1]
			number_of_elem += 1

		median_point[0] /= number_of_elem
		median_point[1] /= number_of_elem
		min_dist = euclidean_distance(median_point, clusters[i][0])
		index_min = 0
		for j in range(number_of_elem):
			if euclidean_distance(median_point, clusters[i][j]) < min_dist:
				min_dist = euclidean_distance(median_point, clusters[i][j])
				index_min = j

		centroids[i] = clusters[i][index_min]

	# print(centroids)
	return centroids


def redistribute_points(sequence_copy, centroids, k):
	temp_clusters = [[]] * k
	for i in range(N):
		min_dist = euclidean_distance(sequence_copy[i], centroids[0])
		index = 0
		for j in range(k):
			if euclidean_distance(sequence_copy[i], centroids[j]) < min_dist:
				min_dist = euclidean_distance(sequence_copy[i], centroids[j])
				index = j

		temp_clusters[index] = temp_clusters[index] + [sequence_copy[i]]

	# print(temp_clusters, len(temp_clusters))
	return temp_clusters


def k_means_alg(sequence, k):
	sequence_copy = sequence
	centroids = rndCenterGenerator()
	# Запис точок до конкретного центру вперше
	clusters = redistribute_points(sequence_copy, centroids, k)
	iterator = 0

	for i in range(50):
		centroids_new = findNewCenters(clusters, centroids)
		# centroids_new = rndCenterGenerator()
		clusters = redistribute_points(sequence_copy, centroids_new, k)
		iterator += 1
		if centroids_new == centroids:
			break

		centroids = centroids_new
	return clusters, iterator, centroids


clusters_fin, iter, centroids = k_means_alg(sequence, k)
# print(clusters_fin, iter)

xy_1 = [[], []]
for i in range(len(clusters_fin[0])):
	xy_1[0].append(clusters_fin[0][i][0])
	xy_1[1].append(clusters_fin[0][i][1])


xy_2 = [[], []]
for i in range(len(clusters_fin[1])):
	xy_2[0].append(clusters_fin[1][i][0])
	xy_2[1].append(clusters_fin[1][i][1])


xy_3 = [[], []]
for i in range(len(clusters_fin[2])):
	xy_3[0].append(clusters_fin[2][i][0])
	xy_3[1].append(clusters_fin[2][i][1])


# xy_4 = [[], []]
# for i in range(len(clusters_fin[3])):
# 	xy_4[0].append(clusters_fin[3][i][0])
# 	xy_4[1].append(clusters_fin[3][i][1])
# #
# #
# xy_5 = [[], []]
# for i in range(len(clusters_fin[4])):
# 	xy_5[0].append(clusters_fin[4][i][0])
# 	xy_5[1].append(clusters_fin[4][i][1])
#
#
# xy_6 = [[], []]
# for i in range(len(clusters_fin[5])):
# 	xy_6[0].append(clusters_fin[5][i][0])
# 	xy_6[1].append(clusters_fin[5][i][1])
#
#
# xy_7 = [[], []]
# for i in range(len(clusters_fin[6])):
# 	xy_7[0].append(clusters_fin[6][i][0])
# 	xy_7[1].append(clusters_fin[6][i][1])


xy = [xy_1, xy_2, xy_3]
#xy = [xy_1, xy_2, xy_3, xy_4, xy_5]
#xy = [xy_1, xy_2, xy_3, xy_4, xy_5, xy_6, xy_7]

plt.figure(dpi=400)
plt.scatter(xy[0][0], xy[0][1], color='blue', alpha=0.5)
plt.scatter(centroids[0][0], centroids[0][1], s=100, marker='x', color='black', linewidths=2)
plt.scatter(xy_2[0], xy_2[1], color='green', alpha=0.5)
plt.scatter(centroids[1][0], centroids[1][1], s=100, marker='x', color='black', linewidths=2)
plt.scatter(xy_3[0], xy_3[1], color='red', alpha=0.5)
plt.scatter(centroids[2][0], centroids[2][1], s=100, marker='x', color='black', linewidths=2)
# plt.scatter(xy_4[0], xy_4[1], color='yellow', alpha=0.5)
# plt.scatter(centroids[3][0], centroids[3][1], color='yellow', s=200, alpha=1)
# plt.scatter(xy_5[0], xy_5[1], color='purple', alpha=0.5)
# plt.scatter(centroids[4][0], centroids[4][1], color='purple', s=200, alpha=1)
# plt.scatter(xy_6[0], xy_6[1], color='black', alpha=0.5)
# plt.scatter(centroids[5][0], centroids[5][1], color='black', s=200, alpha=1)
# plt.scatter(xy_7[0], xy_7[1], color='orange', alpha=0.5)
# plt.scatter(centroids[6][0], centroids[6][1], color='orange', s=200, alpha=1)

# Displaying the plot
plt.grid()
plt.show()


def distance(a, b):
	return euclidean(a, b)


def hierarchical_clustering(data, k):
	distances = pdist(data)
	K = linkage(distances, method='ward')
	clusters = fcluster(K, k, criterion='maxclust')
	return clusters


data = [(random.uniform(0, 1), random.uniform(0, 1)) for _ in range(1000)]
k = 3
hierarchical_clusters = hierarchical_clustering(data, k)


print("\n\nK-means:")
for i in range(k):

	print("Cluster {}: {} points, center at {}".format(i+1, len(clusters_fin[i]), centroids[i]))

print("\n\nHierarchical:")
for i in range(1, k + 1):
	cluster = [data[j] for j in range(len(hierarchical_clusters)) if hierarchical_clusters[j] == i]
	center = (
		sum(point[0] for point in cluster) / len(cluster),
		sum(point[1] for point in cluster) / len(cluster)
	)
	print("Cluster {}: {} points, center at {}".format(i, len(cluster), center))


plt.figure(dpi=400)
for i in range(1, k + 1):
	cluster = [data[j] for j in range(len(hierarchical_clusters)) if hierarchical_clusters[j] == i]
	x = [point[0] for point in cluster]
	y = [point[1] for point in cluster]
	plt.scatter(x, y, label="Cluster {}".format(i))
plt.title("Hierarchical")

plt.show()