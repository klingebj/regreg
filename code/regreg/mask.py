import numpy as np
from scipy import sparse


def adj_from_nii(maskfile,num_time_points,numt=0,numx=1,numy=1,numz=1,regions=None):
    from nipy.io.api import load_image
    from nipy.core.api import Image
    mask = load_image(maskfile)._data
    return adj_from_3dmask(mask=mask,num_time_points=num_time_points,numt=numt,numx=numx,numy=numy,numz=numz,regions=regions)

def adj_from_3dmask(mask,num_time_points,numx=1,numy=1,numz=1,regions=None):
    """
    Construct adjacency array from .nii mask file

    INPUT:

    maskfile: Path to mask file (.nii)

    Other parameters are passed directly to prepare_adj (see that function for docs)

    OUTPUT:

    adj: An array containing adjacency information
    """
    adj = prepare_adj(mask,numt,numx,numy,numz,regions)
    adj = convert_to_array(adj)
    return adj

def prepare_adj(mask, numx=1,numy=1,numz=1,regions=None,return_array=True):
    """
    Return adjacency list, where the voxels are considered neighbors if they
    fall in a ball of radius numt, numx, numy, and numz for time, x position, y
    position, and z position respectively.

    Parameters
    ----------
    X :  (T, N, P, Q, R) shape ndarray
        The first index is trial, the second index is time, the third index is x
        position, the fourth index is y position and the fifth position is z
        position.
    mask : (N, P, Q, R) shape binary ndarray
        The same size as X[0,:,:,:,:] where 1 indicates that the voxel-timepoint
        is included and 0 indicates that it is excluded. NOTE: Usually the mask
        is thought of as a 3-dimensional ndarray, since it is uniform across
        time.
    regions : (N, P, Q, R) shape ndarray
        A multivalued array the same size as the mask that indicates different
        regions in the spatial structure. No adjacency edges will be made across
        region boundaries.
    numt: int XXX NOT IN ARG LIST XXX
        The radius of the "neighborhood ball" in the t direction
    numx : int, optional
        The radius of the "neighborhood ball" in the x direction
    numy : int, optional
        The radius of the "neighborhood ball" in the y direction
    numz : int, optional
        The radius of the "neighborhood ball" in the z direction
    regions :
    return_array : {True, False}, optional

    Returns
    -------
    newX : The matrix X reshaped as a 2-dimensional array for analysis
    adj: The adjacency list associated with newX
    """
    #Create map going from X to predictor vector indices. The entries of
    # this array are -1 if the voxel is not included in the mask, and the 
    # index in the new predictor corresponding to the voxel if the voxel
    # is included in the mask.

    if regions == None:
        regions = np.zeros(mask.shape)
    regions.shape = mask.shape
    reg_values = np.unique(regions)
    
    vmap = np.cumsum(mask).reshape(mask.shape)
    mask = np.bool_(mask.copy())
    vmap[~mask] = -1
    vmap -= 1 # now vmap's values run from 0 to mask.sum()-1

    # Create adjacency list
    
    adj = []

    nx,ny,nz = mask.shape

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if mask[i,j,k]:
                    local_map = vmap[max((i-numx),0):(i+numx+1),
                                     max((j-numy),0):(j+numy+1),
                                     max((k-numz),0):(k+numz+1)]
                    local_reg = regions[max((i-numx),0):(i+numx+1),
                                        max((j-numy),0):(j+numy+1),
                                        max((k-numz),0):(k+numz+1)]
                    region = regions[i,j,k]
                    ind = (local_map>-1)*(local_reg == region)
                    ind = np.bool_(ind)
                    nbrs = np.array(local_map[ind],dtype=int)
                    adj.append(nbrs)



    for i, a in enumerate(adj):
        a[np.equal(a,i)] = -1


    if return_array:
        return convert_to_array(adj)
    else:
        return adj


def create_D(adj):
    """
    Create a matrix D based on the adj data structure
    """

    p, d =  adj.shape
    D = sparse.lil_matrix((np.sum(adj>-1)/2,p))
    count = 0
    for i in range(p):
        for j in range(d):
            nbr = adj[i,j]
            if nbr > i:
                D[count,i] = 1
                D[count,nbr] = -1
                count += 1
    return D

def convert_to_array(adj):
    num_ind = np.max([len(a) for a in adj])
    adjarray = -np.ones((len(adj),num_ind),dtype=np.int)
    for i in range(len(adj)):
        for j in range(len(adj[i])):
            adjarray[i,j] = adj[i][j]
    return adjarray
   
def test_prep(nt=0,nx=1,ny=1,nz=1):
    """
    Let's make this into a proper test...... what should newa, adj be in this case?
    """
    a = np.array(range(1,1+2*3*4*4*4)).reshape((2,3,4,4,4))
    mask = a[0]*0
    mask[:,0,0,0] = 1
    mask[:,1,1,:] = 1
#    print mask[0]
#    print a[0,0]
    newa, adj = prepare_adj(a,mask,nt,nx,ny,nz)
#    print newa[0,0], adj[0], newa[0,adj[0]]
