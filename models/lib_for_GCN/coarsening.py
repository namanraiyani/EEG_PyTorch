import numpy as np
import scipy.sparse

def coarsen(A, levels, self_connections=False):
    graphs, parents = metis(A, levels)
    perms = compute_perm(parents)
    for i, A in enumerate(graphs):
        M, M = A.shape
        if not self_connections:
            A = A.tocoo()
            A.setdiag(0)
        if i < levels:
            A = perm_adjacency(A, perms[i])
        A = A.tocsr()
        A.eliminate_zeros()
        graphs[i] = A
        Mnew, Mnew = A.shape
        print('Layer {0}: M_{0} = |V| = {1} nodes ({2} added),'
              '|E| = {3} edges'.format(i, Mnew, Mnew-M, A.nnz//2))
    return graphs, perms[0] if levels > 0 else None

def metis(W, levels, rid=None):
    N, N = W.shape
    if rid is None:
        rid = np.random.permutation(range(N))
    parents = []
    degree = W.sum(axis=0) - W.diagonal()
    graphs = []
    graphs.append(W)
    for _ in range(levels):
        weights = degree
        weights = np.array(weights).squeeze()
        idx_row, idx_col, val = scipy.sparse.find(W)
        perm = np.argsort(idx_row)
        rr = idx_row[perm]
        cc = idx_col[perm]
        vv = val[perm]
        cluster_id = metis_one_level(rr, cc, vv, rid, weights)
        parents.append(cluster_id)
        nrr = cluster_id[rr]
        ncc = cluster_id[cc]
        nvv = vv
        Nnew = cluster_id.max() + 1
        W = scipy.sparse.csr_matrix((nvv, (nrr, ncc)), shape=(Nnew, Nnew))
        W.eliminate_zeros()
        graphs.append(W)
        N, N = W.shape
        degree = W.sum(axis=0)
        ss = np.array(W.sum(axis=0)).squeeze()
        rid = np.argsort(ss)
    return graphs, parents

def metis_one_level(rr, cc, vv, rid, weights):
    nnz = rr.shape[0]
    N = rr[nnz-1] + 1
    marked = np.zeros(N, bool)
    rowstart = np.zeros(N, np.int32)
    rowlength = np.zeros(N, np.int32)
    cluster_id = np.zeros(N, np.int32)
    oldval = rr[0]
    count = 0
    clustercount = 0
    for ii in range(nnz):
        rowlength[count] = rowlength[count] + 1
        if rr[ii] > oldval:
            oldval = rr[ii]
            rowstart[count+1] = ii
            count = count + 1
    for ii in range(N):
        tid = rid[ii]
        if not marked[tid]:
            wmax = 0.0
            rs = rowstart[tid]
            marked[tid] = True
            bestneighbor = -1
            for jj in range(rowlength[tid]):
                nid = cc[rs+jj]
                if marked[nid]:
                    tval = 0.0
                else:
                    tval = vv[rs+jj] * (1.0/weights[tid] + 1.0/weights[nid])
                if tval > wmax:
                    wmax = tval
                    bestneighbor = nid
            cluster_id[tid] = clustercount
            if bestneighbor > -1:
                cluster_id[bestneighbor] = clustercount
                marked[bestneighbor] = True
            clustercount += 1
    return cluster_id

def compute_perm(parents):
    indices = []
    if len(parents) > 0:
        M_last = max(parents[-1]) + 1
        indices.append(list(range(M_last)))
    for parent in parents[::-1]:
        pool_singeltons = len(parent)
        indices_layer = []
        for i in indices[-1]:
            indices_node = list(np.where(parent == i)[0])
            assert 0 <= len(indices_node) <= 2
            if len(indices_node) == 1:
                indices_node.append(pool_singeltons)
                pool_singeltons += 1
            elif len(indices_node) == 0:
                indices_node.append(pool_singeltons+0)
                indices_node.append(pool_singeltons+1)
                pool_singeltons += 2
            indices_layer.extend(indices_node)
        indices.append(indices_layer)
    return indices[::-1]

def perm_data(x, indices):
    if indices is None:
        return x
    N, M = x.shape
    Mnew = len(indices)
    assert Mnew >= M
    xnew = np.empty((N, Mnew), dtype=x.dtype)
    for i, j in enumerate(indices):
        if j < M:
            xnew[:, i] = x[:, j]
        else:
            xnew[:, i] = np.zeros(N)
    return xnew

def perm_adjacency(A, indices):
    if indices is None:
        return A
    M, M = A.shape
    Mnew = len(indices)
    assert Mnew >= M
    A = A.tocoo()
    if Mnew > M:
        rows = scipy.sparse.coo_matrix((Mnew-M,    M), dtype=np.float32)
        cols = scipy.sparse.coo_matrix((Mnew, Mnew-M), dtype=np.float32)
        A = scipy.sparse.vstack([A, rows])
        A = scipy.sparse.hstack([A, cols])
    perm = np.argsort(indices)
    A.row = np.array(perm)[A.row]
    A.col = np.array(perm)[A.col]
    return A
