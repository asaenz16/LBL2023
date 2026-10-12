"""Compile local thermal-collision templates and estimate ladder resources.

Collision sweeps approximate the Lindblad channels in the small-step limit.
Counts are for an explicit, generic synthesis, not optimized lower bounds.
No coupling-map routing or error correction is included.
"""
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ.setdefault(name,'1')
import json
import time
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from qiskit import QuantumCircuit,transpile
from qiskit.circuit.library import UnitaryGate
from thermal_bath import build_bath,thermal_contact,contact_circuit,channel_matrix
from ladder_tensor import build_ladder,X,Y,Z
from bath_modifications import plaquette_hamiltonian,plaquette_generator


def collision_bath(h,b,duration,weights):
    bath=build_bath(h,beta=b*int(np.log2(len(h))),dt=duration)
    bath.contacts=[thermal_contact(c.label,c.omega,c.lowering,b,duration*weights[int(c.label[1:])]) for c in bath.contacts]
    return bath


def compile_template(name,h,weights,duration=.001):
    bath=collision_bath(h,6.,duration,weights)
    counts={'cx':0,'u3':0,'reset':0,'measure':0};depth=0;per_contact=[];start=time.time()
    for i,contact in enumerate(bath.contacts):
        circuit=contact_circuit(contact,bath.n)
        compiled=transpile(circuit,basis_gates=['u3','cx'],optimization_level=1,seed_transpiler=1200)
        ops=compiled.count_ops()
        for key in counts:counts[key]+=int(ops.get(key,0))
        depth+=compiled.depth();per_contact.append(int(ops.get('cx',0)))
        if (i+1)%100==0:print(name,i+1,'/',len(bath.contacts),'contacts',flush=True)
    return {'name':name,'system_qubits':bath.n,'bath_qubits_serial':1,'contact_duration':duration,
            'contacts':len(bath.contacts),'counts_sum':counts,'depth_sum':depth,
            'cx_per_contact_min':min(per_contact),'cx_per_contact_max':max(per_contact),
            'seconds':time.time()-start}


def main():
    root=Path(__file__).resolve().parents[1]/'results'/'bath-modifications';root.mkdir(parents=True,exist_ok=True)
    delta=float(build_ladder(4).horizontal_delta[1]);templates=[]
    for k,d,name in [(2,2.,'bond_same'),(2,delta,'bond_boundary'),(4,2.,'plaquette_same'),(4,delta,'plaquette_boundary')]:
        path=root/(name+'-circuit.json')
        if path.exists():templates.append(json.loads(path.read_text()));continue
        h=(-.25*(np.kron(X,X)+np.kron(Y,Y)+d*np.kron(Z,Z)) if k==2 else plaquette_hamiltonian(d))
        result=compile_template(name,h,[1.]*2 if k==2 else [1.5]*4)
        path.write_text(json.dumps(result,indent=2)+'\n');templates.append(result)
        print(result,flush=True)
    h=-.25*(np.kron(X,X)+np.kron(Y,Y)+delta*np.kron(Z,Z))
    circuit=QuantumCircuit(2);circuit.append(UnitaryGate(expm(-1j*.025*h)),[0,1])
    coherent=transpile(circuit,basis_gates=['u3','cx'],optimization_level=1,seed_transpiler=1200)
    costs={r['name']:r for r in templates};rows=[]
    for L in [4,8,12,16]:
        bond_cx=4*L*costs['bond_same']['counts_sum']['cx']+(L-2)*costs['bond_boundary']['counts_sum']['cx']
        plaque_cx=L*costs['plaquette_same']['counts_sum']['cx']+(L//2-1)*costs['plaquette_boundary']['counts_sum']['cx']
        bond_contacts=4*L*costs['bond_same']['contacts']+(L-2)*costs['bond_boundary']['contacts']
        plaque_contacts=L*costs['plaquette_same']['contacts']+(L//2-1)*costs['plaquette_boundary']['contacts']
        rows.append({'width':L,'system_qubits':2*L,'bond_cx_per_collision_sweep_estimate':bond_cx,
                     'plaquette_cx_per_collision_sweep_estimate':plaque_cx,'bond_contacts':bond_contacts,
                     'plaquette_contacts':plaque_contacts,
                     'coherent_added_cx_per_strang_step':(5*L-2)*int(coherent.count_ops().get('cx',0))})
    report={'templates':templates,'ladder_estimates':rows,'coherent_two_spin_block_counts':dict(coherent.count_ops()),
            'compilation':{'basis':['u3','cx'],'optimization_level':1,'seed':1200,'routing':None},
            'limitations':['Resource estimates use local templates at the 2x4 contrast coupling; larger-size and boundary proposal weights can change synthesis constants.',
              'Two half-steps and one full-step use the same generic template count estimate.',
              'Collision sweeps approximate, and are not identical to, the exponential dissipative channels simulated in the study.',
              'No optimized synthesis, routing, error correction, required contact subdivision, shot repetition, or mixing-time guarantee included.']}
    (root/'circuit-resources.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
