import numpy as np
from qiskit_aer import Aer
from qiskit import QuantumCircuit, transpile, assemble
from scipy.linalg import eigvals
import matplotlib.pyplot as plt


def effective_hamiltonian(theta, gamma):
   
    sigma_x = np.array([[0, 1], [1, 0]])  
    sigma_z = np.array([[1, 0], [0, -1]]) 
    I = np.eye(2)
    H = (theta / 2) * sigma_x + (1j * gamma / 2) * (sigma_z - I)
    return H

theta = 1.0  
gammas = np.linspace(0.01, 2.0, 200)  
eigenvalues = []

for gamma in gammas:
    H = effective_hamiltonian(theta, gamma)
    eigenvalues.append(eigvals(H))


Re = np.array([np.real(ev) for ev in eigenvalues])
Im = np.array([np.imag(ev) for ev in eigenvalues])

plt.figure(figsize=(12, 6))


plt.subplot(1, 2, 1)
for i in range(Re.shape[1]):  
    plt.plot(gammas, Re[:, i], label=f"Re(eigenvalue {i+1})")
plt.axvline(x=1.0, color='red', linestyle='dashed', label="Exceptional Point")
plt.title("Real Parts of Eigenvalues")
plt.xlabel("Gamma (Γ)")
plt.ylabel("Real(Eigenvalue)")
plt.legend()
plt.grid()


plt.subplot(1, 2, 2)
for i in range(Im.shape[1]):
    plt.plot(gammas, Im[:, i], label=f"Im(eigenvalue {i+1})")
plt.axvline(x=1.0, color='red', linestyle='dashed', label="Exceptional Point")
plt.title("Imaginary Parts of Eigenvalues")
plt.xlabel("Gamma (Γ)")
plt.ylabel("Imaginary(Eigenvalue)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()


def create_non_hermitian_circuit(theta, gamma, cycles):

    qc = QuantumCircuit(2, 1)  
    qc.initialize([1, 0], 1)  
    for _ in range(cycles):
        qc.rx(theta, 0)
        qc.crx(2 * np.arcsin(np.sqrt(gamma / 2)), 0, 1)
        qc.measure(1, 0)
        qc.reset(1)  
    return qc
gamma_test = 1.0 
qc = create_non_hermitian_circuit(theta, gamma_test, cycles=3)
print("\nQuantum Circuit:\n")
print(qc)
sim = Aer.get_backend('qasm_simulator')
tqc = transpile(qc, sim)
result = sim.run(tqc, shots=1000).result()
count = result.get_counts()
print("\nMeasurement Results:", count)
