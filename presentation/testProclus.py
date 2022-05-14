class testProclus:
    '''\n
    A, M, D: Output A, M and D obtained after performing proClus.
    X: Pre processed data as pandas DataFrame'''
    def __init__(self, A, M, D, X):
        self.A = A
        self.M = M
        self.D = D
        self.X = X
    def assignCluster(self, point, metric = "manhattan"):
        '''
        point: Data point after Preprocessing as pandas DataFrame. 
        metric = Distance metric. See: sklearn.neighbors.DistanceMetric'''
        M = list(self.M)
        D = list(self.D)
        from sklearn.neighbors import DistanceMetric
        dist = DistanceMetric.get_metric(metric)
        distance = []
        for i in range(len(M)):
            current_M = M[i]
            current_D = list(D[i])
            pt_x = point[current_D]
            pt_m = self.X.iloc[current_M, :]
            pt_m = pt_m[current_D]
            current_distance = dist.pairwise(
                [
                    pt_x,
                    pt_m
                ]
            )
            distance.append(current_distance[0,1]/len(current_D))
        minpos = distance.index(min(distance))
        return (M[minpos])
