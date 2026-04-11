import numpy as np
import os

def solve_elastostatics(E_macro=50e9, E_micro=10e9, length=1.0, load=1e6, nx=50):
    x = np.linspace(0, length, nx+1)
    dx = x[1] - x[0]
    
    alpha = 0.3
    E_eff = alpha * E_micro + (1-alpha) * E_macro
    
    K = np.zeros((nx+1, nx+1))
    for i in range(nx):
        k_elem = E_eff / dx  # Fixed: no *1.0 causing nan
        Ke = np.array([[k_elem, -k_elem], [-k_elem, k_elem]])
        K[i:i+2, i:i+2] += Ke  # Fixed: np.array broadcasts correctly
    
    F = np.zeros(nx+1)
    F[-1] = load
    
    # Fix left end
    K[0, :] = 0
    K[0, 0] = 1
    F[0] = 0
    
    u = np.linalg.solve(K, F)
    
    stress = E_eff * np.diff(u) / dx
    stress = np.insert(stress, 0, 0)  # Pad front
    
    os.makedirs('output', exist_ok=True)
    np.savetxt('output/x.txt', x)
    np.savetxt('output/displacement.txt', u)
    np.savetxt('output/stress.txt', stress)
    
    return {'x': x, 'u': u, 'stress': stress, 'E_eff': E_eff}

if __name__ == "__main__":
    results = solve_elastostatics()
    print("Solver done! Files in output/")
    print(f"E_eff = {results['E_eff']/1e9:.1f} GPa")
    print(f"Max displacement: {np.max(np.abs(results['u']))*1e6:.2f} μm")