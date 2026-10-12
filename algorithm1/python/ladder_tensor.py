"""2 x L image XXZ ladders as rung MPS/MPO, with L=4N for the study.

Each tensor site holds a column's top and bottom spins in that order,
physical basis |00>,|01>,|10>,|11>. Public observables are returned as
(row,column) arrays, matching the paper's row-major image convention.
The Hamiltonian and unique-edge sample-sigma normalization match algorithm1.
"""
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.linalg import expm
import quimb.tensor as qtn

# Predictable single-process contractions; avoid automatic parallel path search.
qtn.set_contract_strategy("greedy")

I2 = np.eye(2)
X = np.array([[0,1],[1,0.]])
Y = np.array([[0,-1j],[1j,0]])
Z = np.diag([1.,-1.])
I4 = np.eye(4)
OPS = {axis:(np.kron(op,I2),np.kron(I2,op))
       for axis,op in zip("XYZ",[X,Y,Z])}


@dataclass
class Ladder:
    width: int
    image: np.ndarray
    sigma: float
    normalization: str
    kappa: float
    horizontal_delta: np.ndarray
    h_rung: np.ndarray
    h_bonds: dict
    mpo: object
    local_ham: object

    @property
    def n_spins(self):
        return 2*self.width


def build_ladder(width, kappa=2., normalization="sample"):
    if not isinstance(width,Integral) or width < 4 or width % 4:
        raise ValueError("The strip width must be a positive multiple of four.")
    if not np.isfinite(kappa) or kappa < 0:
        raise ValueError("kappa must be finite and nonnegative.")
    colors = 255*((np.arange(width)//2)%2)
    image = np.stack([colors,colors])
    contrasts = np.r_[np.zeros(width),np.tile(np.abs(np.diff(colors)),2)]
    if normalization == "sample":
        sigma = float(np.std(contrasts,ddof=1))
    elif normalization == "bulk":
        # Infinite-strip fraction of nonzero contrasts is 1/3.
        sigma = 255*np.sqrt(2)/3
    else:
        raise ValueError("normalization must be 'sample' or 'bulk'.")
    delta = 2-kappa*np.abs(np.diff(colors))/sigma
    h_rung = np.real_if_close(-.25*(np.kron(X,X)+np.kron(Y,Y)+2*np.kron(Z,Z)))
    operators = [OPS[axis][row] for row in range(2) for axis in "XYZ"]
    h_bonds = {}
    tensors = []
    for col in range(width):
        coefficients = -.25*np.tile([1.,1.,delta[col] if col < width-1 else 0.],2)
        w = np.zeros((8,8,4,4),complex)
        w[0,0] = I4; w[-1,-1] = I4; w[-1,0] = h_rung
        for k,(coefficient,op) in enumerate(zip(coefficients,operators),1):
            w[-1,k] = coefficient*op
            w[k,0] = op
        tensors.append(w[-1] if col == 0 else w[:,0] if col == width-1 else w)
        if col < width-1:
            h_bonds[col,col+1] = np.real_if_close(sum(
                coefficient*np.kron(op,op) for coefficient,op in zip(coefficients,operators)))
    mpo = qtn.MatrixProductOperator(tensors)
    # XX and YY products make the full MPO real, but its factorization has Y.
    return Ladder(width,image,sigma,normalization,float(kappa),delta,h_rung,h_bonds,
                  mpo,qtn.LocalHam1D(width,H2=h_bonds,H1=h_rung,cyclic=False))


def domain_product(ladder):
    return qtn.MPS_product_state([np.eye(4)[0 if color == 0 else 3]
                                  for color in ladder.image[0]])


def global_flip(psi):
    result = psi.copy()
    for col in range(psi.L):
        result.gate_(np.kron(X,X),col,contract=True)
    return result


def even_projection(psi):
    result = psi.add_MPS(global_flip(psi))
    result.compress(cutoff=1e-12)
    result.normalize()
    return result


def energy(psi, ladder):
    arrays=rung_arrays(psi)
    operators=list(ladder.mpo.arrays)
    operators[0]=operators[0][None]; operators[-1]=operators[-1][:,None]
    environment=np.ones((1,1,1),complex)
    for a,w in zip(arrays,operators):
        # Explicit order avoids an O(chi^4) intermediate from generic planning.
        tmp=np.tensordot(environment,a,axes=([2],[0]))  # a,w,d,t
        tmp=np.tensordot(tmp,w,axes=([1,3],[0,3])).transpose(0,2,1,3)  # a,x,d,s
        environment=np.tensordot(a.conj(),tmp,axes=([0,2],[0,3]))  # b,x,d
    return float(np.real(environment[0,0,0]))


def ground_state(ladder, bond_dims=(16,32,64), cutoff=1e-11, max_sweeps=16,
                 tolerance=1e-9, initial=None):
    initial = even_projection(domain_product(ladder)) if initial is None else initial
    dmrg = qtn.DMRG2(ladder.mpo,bond_dims=bond_dims,cutoffs=cutoff,p0=initial)
    dmrg.opts["local_eig_tol"] = 1e-10
    converged = dmrg.solve(tol=tolerance,max_sweeps=max_sweeps,verbosity=0)
    psi = even_projection(dmrg.state)
    # Projection restores global spin-flip symmetry without changing H.
    e = energy(psi,ladder)
    hpsi = ladder.mpo.apply(psi,compress=False)
    env=np.ones((1,1),complex)
    for a in rung_arrays(hpsi): env=_left(env,a)
    variance = max(0.,float(np.real(env[0,0]))-e*e)
    return psi,{"converged":bool(converged),"energies":[float(np.real(x)) for x in dmrg.energies],
                "energy":e,"variance":variance,"max_bond":psi.max_bond(),
                "bond_dims":list(bond_dims),"cutoff":cutoff,
                "symmetry":"even projection under global X flip"}


def rung_arrays(psi):
    state = psi.copy()
    state.permute_arrays("lrp")
    arrays = list(state.arrays)
    arrays[0] = arrays[0][None,:,:]
    arrays[-1] = arrays[-1][:,None,:]
    return arrays


def _left(environment, tensor, operator=None):
    # E[b,d] = sum E[a,c] A[a,b,s] conj(A[c,d,t]) O[t,s].
    if operator is None:
        return sum(tensor[:,:,s].T@environment@tensor[:,:,s].conj() for s in range(4))
    transformed = np.einsum("abs,ts->abt",tensor,operator)
    return sum(transformed[:,:,t].T@environment@tensor[:,:,t].conj() for t in range(4))


def _right(environment, tensor, operator=None):
    if operator is None:
        return sum(tensor[:,:,s]@environment@tensor[:,:,s].conj().T for s in range(4))
    transformed = np.einsum("abs,ts->abt",tensor,operator)
    return sum(transformed[:,:,t]@environment@tensor[:,:,t].conj().T for t in range(4))


def seed_profile(psi, seed=(0,0)):
    """Raw and connected <Sz_seed Sz_p>, means, without dense conversion."""
    row,col = seed
    if row not in (0,1) or not 0 <= col < psi.L:
        raise ValueError("Invalid seed coordinate.")
    arrays = rung_arrays(psi); L = len(arrays)
    left = [np.ones((1,1),complex)]
    for a in arrays: left.append(_left(left[-1],a))
    right = [None]*(L+1); right[L] = np.ones((1,1),complex)
    for j in reversed(range(L)): right[j] = _right(right[j+1],arrays[j])
    norm = float(left[-1][0,0].real)
    means = np.empty((2,L)); gamma = np.empty((2,L))
    spin = [op/2 for op in OPS["Z"]]; seed_op = spin[row]
    for j in range(L):
        for r in range(2):
            means[r,j] = np.real(np.sum(_left(left[j],arrays[j],spin[r])*right[j+1]))/norm
    for r in range(2):
        gamma[r,col] = np.real(np.sum(_left(left[col],arrays[col],seed_op@spin[r])*right[col+1]))/norm
    marked = _left(left[col],arrays[col],seed_op)
    for j in range(col+1,L):
        for r in range(2):
            gamma[r,j] = np.real(np.sum(_left(marked,arrays[j],spin[r])*right[j+1]))/norm
        marked = _left(marked,arrays[j])
    marked = _right(right[col+1],arrays[col],seed_op)
    for j in reversed(range(col)):
        for r in range(2):
            gamma[r,j] = np.real(np.sum(left[j]*_right(marked,arrays[j],spin[r])))/norm
        marked = _right(marked,arrays[j])
    return {"gamma":gamma,"connected":gamma-means[row,col]*means,"means":means,"norm":norm}


def rotate(psi, angle, axis="X", signs=None):
    """One layer of physical rotations exp(-i angle P_j/2), not stochastic noise."""
    if axis not in OPS or not np.isfinite(angle):
        raise ValueError("Require X/Y/Z and a finite angle.")
    signs = np.ones((2,psi.L)) if signs is None else np.asarray(signs)
    if signs.shape != (2,psi.L):
        raise ValueError("Rotation signs must have shape (2,width).")
    state = psi.copy()
    for col in range(psi.L):
        generator = signs[0,col]*OPS[axis][0]+signs[1,col]*OPS[axis][1]
        state.gate_(expm(-.5j*angle*generator),col,contract=True)
    return state


def save_mps(path, psi):
    arrays = rung_arrays(psi)
    arrays[0] = arrays[0][0]; arrays[-1] = arrays[-1][:,0]
    np.savez_compressed(path,**{f"site_{j}":a for j,a in enumerate(arrays)})


def load_mps(path):
    with np.load(path) as data:
        arrays = [data[f"site_{j}"] for j in range(len(data.files))]
    return qtn.MatrixProductState(arrays,shape="lrp")


def dense_row_hamiltonian(ladder):
    """Independent bit-basis benchmark, limited to the 2x4 image."""
    if ladder.width != 4:
        raise ValueError("Dense validation is only for 2x4.")
    L = ladder.width; n=2*L; basis=np.arange(2**n); h=np.zeros((2**n,2**n))
    edges = [(c,L+c,2.) for c in range(L)]
    edges += [(r*L+c,r*L+c+1,ladder.horizontal_delta[c]) for r in range(2) for c in range(L-1)]
    for i,j,delta in edges:
        a,b=1<<(n-1-i),1<<(n-1-j)
        opposite=((basis&a)!=0)!=((basis&b)!=0)
        h[basis,basis] -= delta*(1-2*opposite.astype(int))/4
        selected=basis[opposite]
        h[selected^a^b,selected] -= .5
    return h


def rung_to_row_indices(width):
    if width > 4:
        raise ValueError("Dense index conversion is only used up to 2x4.")
    basis=np.arange(4**width); row_indices=np.zeros_like(basis)
    for col in range(width):
        top=(basis>>(2*width-1-2*col))&1; bottom=(basis>>(2*width-2-2*col))&1
        row_indices |= top<<(2*width-1-col)
        row_indices |= bottom<<(width-1-col)
    return row_indices
