# import random
#
# import matplotlib.pyplot as plt
#
#
# def f(x):
# 	x = 5
# 	return x
#
#
# x = 2
# x = f(x)
# print(x)
#
#
# arr = [[]] * 3
# arr2 = [[], [], []]
# arr[1] = arr[1] + [2]
# arr2[1].extend([2])
# print(arr)
# print(arr2)
#
#
#
#
#
# xy = [[1, 2], [3, 4]]
#
# for i in range(2):
# 	plt.scatter(xy[i][0], xy[i][1], color='red', alpha=0.5)
#
# # Displaying the plot
# plt.grid()
# plt.show()
#


#
# # spectral clustering
# from numpy import unique
# from numpy import where
# from sklearn.datasets import make_classification
# from sklearn.cluster import SpectralClustering
# from matplotlib import pyplot
# # define dataset
# X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
# # define the model
# model = SpectralClustering(n_clusters=3)
# # fit model and predict clusters
# yhat = model.fit_predict(X)
# # retrieve unique clusters
# clusters = unique(yhat)
# # create scatter plot for samples from each cluster
# for cluster in clusters:
# 	# get row indexes for samples with this cluster
# 	row_ix = where(yhat == cluster)
# 	# create scatter of these samples
# 	pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# # show the plot
# pyplot.show()





# import random
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
#
#
# sequence = []
# N = 1000
# k = 7
#
# for i in range(N):
#     temp = []
#     x = random.uniform(0, 1)
#     temp.append(x)
#     y = random.uniform(0, 1)
#     temp.append(y)
#     sequence.append(temp)
#
#
# def euclidean_distance(arr_center, arr):
#     return np.sqrt((arr_center[0] - arr[0]) ** 2 + (arr_center[1] - arr[1]) ** 2)
#
#
# def findNewCenters(clusters, centroids):
#     # Finding new centroids
#     for i in range(k):
#         median_point = [0, 0]
#         number_of_elem = 0
#         for j in range(len(clusters[i])):
#             median_point[0] += clusters[i][j][0]
#             median_point[1] += clusters[i][j][1]
#             number_of_elem += 1
#
#         median_point[0] /= number_of_elem
#         median_point[1] /= number_of_elem
#         min_dist = euclidean_distance(median_point, clusters[i][0])
#         index_min = 0
#         for j in range(number_of_elem):
#             if euclidean_distance(median_point, clusters[i][j]) < min_dist:
#                 min_dist = euclidean_distance(median_point, clusters[i][j])
#                 index_min = j
#
#         centroids[i] = clusters[i][index_min]
#
#     print(centroids)
#     return centroids
#
#
# def redistribute_points(sequence_copy, centroids, k):
#     temp_clusters = [[] for _ in range(k)]
#     for i in range(N):
#         min_dist = euclidean_distance(sequence_copy[i], centroids[0])
#         index = 0
#         for j in range(k):
#             if euclidean_distance(sequence_copy[i], centroids[j]) < min_dist:
#                 min_dist = euclidean_distance(sequence_copy[i], centroids[j])
#                 index = j
#
#         temp_clusters[index].append(sequence_copy[i])
#
#     print(temp_clusters, len(temp_clusters))
#     return temp_clusters
#
#
# def fuzzy_c_means_alg(sequence, k):
#     sequence_copy = sequence
#     centroids = np.array(rndCenterGenerator())
#     clusters = redistribute_points(sequence_copy, centroids, k)
#     iterator = 0
#     m = 2  # fuzziness parameter
#
#     while True:
#         centroids_new = findNewCenters(clusters, centroids)
#         clusters = redistribute_points(sequence_copy, centroids_new, k)
#
#         # Check for convergence
#         if np.linalg.norm(centroids_new - centroids) < 1e-4:
#             break
#
#         centroids = centroids_new
#         iterator += 1
#
#     return clusters, iterator, centroids
#
#
# def rndCenterGenerator():
#     temp_centroids = []
#     rndForCenter = random.sample(range(N), k)
#     for i in rndForCenter:
#         temp_centroids.append(sequence[i])
#
#     print(temp_centroids)
#     return temp_centroids
#
#
# clusters_fin, iter, centroids = fuzzy_c_means_alg(sequence, k)
# print(clusters_fin, iter)
#
# for i in range(k):
#     xy_i = [[], []]
#     for j in range(len(clusters_fin[i])):
#         xy_i[0].append(clusters_fin[i][j][0])
#         xy_i[1].append(clusters_fin[i][j][1])
#     plt.scatter(xy_i[0], xy_i[1], alpha=0.5)
#
# plt.show()


import math
import collections
import random
import copy
import pylab


try:
    import psyco

    psyco.full()
except ImportError:
    pass

FLOAT_MAX = 1e100


class Point:
    __slots__ = ["x", "y", "group", "membership"]

    def __init__(self, clusterCenterNumber, x=0, y=0, group=0):
        self.x, self.y, self.group = x, y, group
        self.membership = [0.0 for _ in range(clusterCenterNumber)]


def generatePoints(pointsNumber, radius, clusterCenterNumber):
    points = [Point(clusterCenterNumber) for _ in range(2 * pointsNumber)]
    count = 0
    for point in points:
        count += 1
        r = random.random() * radius
        angle = random.random() * 2 * math.pi
        point.x = r * math.cos(angle)
        point.y = r * math.sin(angle)
        if count == pointsNumber - 1:
            break
    for index in range(pointsNumber, 2 * pointsNumber):
        points[index].x = 2 * radius * random.random() - radius
        points[index].y = 2 * radius * random.random() - radius
    return points

# print(generatePoints(10, 2, 2)[0].group)


def solveDistanceBetweenPoints(pointA, pointB):
    return (pointA.x - pointB.x) * (pointA.x - pointB.x) + (pointA.y - pointB.y) * (pointA.y - pointB.y)


def getNearestCenter(point, clusterCenterGroup):
    minIndex = point.group
    minDistance = FLOAT_MAX
    for index, center in enumerate(clusterCenterGroup):
        distance = solveDistanceBetweenPoints(point, center)
        if (distance < minDistance):
            minDistance = distance
            minIndex = index
    return (minIndex, minDistance)


def kMeansPlusPlus(points, clusterCenterGroup):
    clusterCenterGroup[0] = copy.copy(random.choice(points))
    distanceGroup = [0.0 for _ in range(len(points))]
    sum = 0.0
    for index in range(1, len(clusterCenterGroup)):
        for i, point in enumerate(points):
            distanceGroup[i] = getNearestCenter(point, clusterCenterGroup[:index])[1]
            sum += distanceGroup[i]
        sum *= random.random()
        for i, distance in enumerate(distanceGroup):
            sum -= distance;
            if sum < 0:
                clusterCenterGroup[index] = copy.copy(points[i])
                break
    return


def fuzzyCMeansClustering(points, clusterCenterNumber, weight):
    clusterCenterGroup = [Point(clusterCenterNumber) for _ in range(clusterCenterNumber)]
    kMeansPlusPlus(points, clusterCenterGroup)
    clusterCenterTrace = [[clusterCenter] for clusterCenter in clusterCenterGroup]
    tolerableError, currentError = 1.0, FLOAT_MAX
    while currentError >= tolerableError:
        for point in points:
            getSingleMembership(point, clusterCenterGroup, weight)
        currentCenterGroup = [Point(clusterCenterNumber) for _ in range(clusterCenterNumber)]
        for centerIndex, center in enumerate(currentCenterGroup):
            upperSumX, upperSumY, lowerSum = 0.0, 0.0, 0.0
            for point in points:
                membershipWeight = pow(point.membership[centerIndex], weight)
                upperSumX += point.x * membershipWeight
                upperSumY += point.y * membershipWeight
                lowerSum += membershipWeight
            center.x = upperSumX / lowerSum
            center.y = upperSumY / lowerSum
        # update cluster center trace
        currentError = 0.0
        for index, singleTrace in enumerate(clusterCenterTrace):
            singleTrace.append(currentCenterGroup[index])
            currentError += solveDistanceBetweenPoints(singleTrace[-1], singleTrace[-2])
            clusterCenterGroup[index] = copy.copy(currentCenterGroup[index])
    for point in points:
        maxIndex, maxMembership = 0, 0.0
        for index, singleMembership in enumerate(point.membership):
            if singleMembership > maxMembership:
                maxMembership = singleMembership
                maxIndex = index
        point.group = maxIndex
    return clusterCenterGroup, clusterCenterTrace


def getSingleMembership(point, clusterCenterGroup, weight):
    distanceFromPoint2ClusterCenterGroup = [solveDistanceBetweenPoints(point, clusterCenterGroup[index]) for index in
                                            range(len(clusterCenterGroup))]
    for centerIndex, singleMembership in enumerate(point.membership):
        sum = 0.0
        isCoincide = [False, 0]
        for index, distance in enumerate(distanceFromPoint2ClusterCenterGroup):
            if distance == 0:
                isCoincide[0] = True
                isCoincide[1] = index
                break
            sum += pow(float(distanceFromPoint2ClusterCenterGroup[centerIndex] / distance), 1.0 / (weight - 1.0))
        if isCoincide[0]:
            if isCoincide[1] == centerIndex:
                point.membership[centerIndex] = 1.0
            else:
                point.membership[centerIndex] = 0.0
        else:
            point.membership[centerIndex] = 1.0 / sum


def showClusterAnalysisResults(points, clusterCenterTrace):
    colorStore = ['or', 'og', 'ob', 'oc', 'om', 'oy', 'ok']
    pylab.figure(figsize=(9, 9), dpi=80)
    for point in points:
        color = ''
        if point.group >= len(colorStore):
            color = colorStore[-1]
        else:
            color = colorStore[point.group]
        pylab.plot(point.x, point.y, color)
    for singleTrace in clusterCenterTrace:
        pylab.plot([center.x for center in singleTrace], [center.y for center in singleTrace], 'k')
    pylab.show()


def main():
    clusterCenterNumber = 5
    pointsNumber = 1000
    radius = 0.5
    weight = 2
    points = generatePoints(pointsNumber, radius, clusterCenterNumber)
    _, clusterCenterTrace = fuzzyCMeansClustering(points, clusterCenterNumber, weight)
    showClusterAnalysisResults(points, clusterCenterTrace)


main()



