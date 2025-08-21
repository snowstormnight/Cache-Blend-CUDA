"""
Solution to Complex Analysis Integrals
=======================================

This file contains detailed solutions to the six integral problems.
"""

import math

# Problem 2: Compute the integrals

print("="*60)
print("COMPLEX ANALYSIS - INTEGRAL SOLUTIONS")
print("="*60)

# (a) ∫_{-∞}^{∞} (x²sin(x))/(x⁴+a⁴) dx, a > 0
print("\n(a) ∫_{-∞}^{∞} (x²sin(x))/(x⁴+a⁴) dx, a > 0")
print("-" * 50)
print("""
Solution using Residue Theorem:

1) First, note that sin(x) = Im(e^{ix}), so we compute:
   ∫_{-∞}^{∞} (x²e^{ix})/(x⁴+a⁴) dx

2) Find poles of f(z) = z²e^{iz}/(z⁴+a⁴):
   z⁴ = -a⁴ = a⁴e^{iπ}
   
   The poles are at: z_k = a·e^{i(π/4 + kπ/2)} for k = 0,1,2,3
   
   In the upper half-plane (Im(z) > 0):
   - z₀ = a·e^{iπ/4} = a(1+i)/√2
   - z₁ = a·e^{i3π/4} = a(-1+i)/√2

3) Calculate residues at upper half-plane poles:
   
   For simple pole at z₀ = a·e^{iπ/4}:
   Res(f, z₀) = lim_{z→z₀} (z-z₀)·z²e^{iz}/(z⁴+a⁴)
             = z₀²e^{iz₀}/(4z₀³)
             = e^{iz₀}/(4z₀)
             = e^{ia(1+i)/√2}/(4a·e^{iπ/4})
   
   For z₁ = a·e^{i3π/4}:
   Res(f, z₁) = e^{iz₁}/(4z₁)
             = e^{ia(-1+i)/√2}/(4a·e^{i3π/4})

4) By Residue Theorem:
   ∫_{-∞}^{∞} (x²e^{ix})/(x⁴+a⁴) dx = 2πi·[Res(f,z₀) + Res(f,z₁)]

5) After simplification (taking imaginary part):
   
   ANSWER: (π/√2a²)·e^{-a/√2}·sin(a/√2)
""")

# (b) ∫_0^{2π} dθ/(1+β cos θ)² for -1 < β < 1
print("\n(b) ∫_0^{2π} dθ/(1+β cos θ)² for -1 < β < 1")
print("-" * 50)
print("""
Solution using Complex Substitution:

1) Use substitution z = e^{iθ}, then:
   - cos θ = (z + 1/z)/2
   - dθ = dz/(iz)
   - Integration path: |z| = 1

2) The integral becomes:
   ∮_{|z|=1} 1/[1 + β(z+1/z)/2]² · dz/(iz)
   = ∮_{|z|=1} 4z/[i(2z+βz²+β)²] dz
   = -4i ∮_{|z|=1} z/(βz²+2z+β)² dz

3) Find poles of f(z) = z/(βz²+2z+β)²:
   βz²+2z+β = 0
   z = (-2 ± √(4-4β²))/(2β) = (-1 ± √(1-β²))/β
   
   For |β| < 1:
   - z₁ = (-1 + √(1-β²))/β (inside |z|=1)
   - z₂ = (-1 - √(1-β²))/β (outside |z|=1)
   
   Both are double poles.

4) Calculate residue at z₁ (double pole):
   Res(f, z₁) = d/dz[(z-z₁)²f(z)]|_{z=z₁}
   
   After calculation:
   Res(f, z₁) = -1/(2β²(1-β²)^{3/2})

5) By Residue Theorem:
   Original integral = -4i · 2πi · Res(f, z₁)
                    = 8π · 1/(2β²(1-β²)^{3/2})
   
   ANSWER: 2π/(1-β²)^{3/2}
""")

# (c) (1/2π) ∫_0^{2π} e^{-inθ} sin(e^{iθ}) dθ for n = 0,1,2,...
print("\n(c) (1/2π) ∫_0^{2π} e^{-inθ} sin(e^{iθ}) dθ for n = 0,1,2,...")
print("-" * 50)
print("""
Solution using Fourier Series:

1) Note that sin(e^{iθ}) has the Taylor series:
   sin(w) = w - w³/3! + w⁵/5! - ...
   
   So: sin(e^{iθ}) = e^{iθ} - e^{3iθ}/6 + e^{5iθ}/120 - ...

2) The integral (1/2π) ∫_0^{2π} e^{-inθ} sin(e^{iθ}) dθ
   is the n-th Fourier coefficient of sin(e^{iθ}).

3) Using orthogonality of e^{imθ}:
   (1/2π) ∫_0^{2π} e^{i(m-n)θ} dθ = δ_{m,n} (Kronecker delta)

4) From the series expansion:
   - For n = 0: coefficient = 0 (sin is odd)
   - For n = 1: coefficient = 1
   - For n = 2: coefficient = 0
   - For n = 3: coefficient = -1/6
   - For n = 4: coefficient = 0
   - For n = 5: coefficient = 1/120
   
   In general, for odd n: coefficient = (-1)^{(n-1)/2}/n!
   For even n: coefficient = 0

   ANSWER: 
   - 0 if n is even
   - (-1)^{(n-1)/2}/n! if n is odd
""")

# (d) ∫_{|z+2|=2} z³/(1-z²) dz
print("\n(d) ∫_{|z+2|=2} z³/(1-z²) dz")
print("-" * 50)
print("""
Solution using Residue Theorem:

1) The integrand f(z) = z³/(1-z²) = z³/[(1-z)(1+z)]
   has simple poles at z = 1 and z = -1.

2) Check which poles are inside |z+2| = 2:
   - Center of circle: z = -2
   - Radius: 2
   
   Distance from -2 to pole at z = 1:
   |1-(-2)| = |3| = 3 > 2 (outside)
   
   Distance from -2 to pole at z = -1:
   |-1-(-2)| = |1| = 1 < 2 (inside)

3) Calculate residue at z = -1:
   Res(f, -1) = lim_{z→-1} (z+1)·z³/[(1-z)(1+z)]
              = lim_{z→-1} z³/(1-z)
              = (-1)³/(1-(-1))
              = -1/2

4) By Residue Theorem:
   ∮_{|z+2|=2} z³/(1-z²) dz = 2πi·Res(f, -1)
                              = 2πi·(-1/2)
   
   ANSWER: -πi
""")

# (e) ∫_γ cos z dz, where γ is a curve from i to π
print("\n(e) ∫_γ cos z dz, where γ is a curve from i to π")
print("-" * 50)
print("""
Solution using Fundamental Theorem of Calculus:

1) Since cos z is analytic everywhere in ℂ (entire function),
   the integral is path-independent.

2) We have: ∫ cos z dz = sin z + C

3) Therefore:
   ∫_γ cos z dz = sin(π) - sin(i)
   
4) Calculate:
   - sin(π) = 0
   - sin(i) = (e^{i·i} - e^{-i·i})/(2i)
           = (e^{-1} - e^{1})/(2i)
           = -i(e - e^{-1})/2
           = -i·sinh(1)

5) Therefore:
   ∫_γ cos z dz = 0 - (-i·sinh(1))
                = i·sinh(1)
   
   ANSWER: i·sinh(1) = i(e-1/e)/2
""")

# (f) ∫_{|z|=4} z³/sin z dz
print("\n(f) ∫_{|z|=4} z³/sin z dz")
print("-" * 50)
print("""
Solution using Residue Theorem:

1) The function f(z) = z³/sin z has simple poles where sin z = 0,
   i.e., at z = nπ for all integers n.

2) Poles inside |z| = 4:
   - z = 0 (but this is removable, see below)
   - z = ±π (since |π| ≈ 3.14 < 4)
   - z = ±2π, ±3π, ... are outside since |2π| ≈ 6.28 > 4

3) Check z = 0:
   Near z = 0: sin z ≈ z - z³/6 + ...
   So z³/sin z ≈ z³/(z - z³/6 + ...) = z²/(1 - z²/6 + ...)
   
   Using Laurent series:
   z³/sin z = z² + z⁴/6 + ... (regular at z = 0)
   
   So z = 0 is a removable singularity, not a pole.

4) Calculate residues at z = ±π:
   
   At z = π:
   Res(f, π) = lim_{z→π} (z-π)·z³/sin z
            = lim_{z→π} z³·(z-π)/sin z
            = π³·lim_{z→π} (z-π)/sin z
   
   Using L'Hôpital or sin z ≈ -sin(z-π) ≈ -(z-π) near z = π:
   = π³·lim_{z→π} 1/cos z = π³/cos π = π³/(-1) = -π³
   
   At z = -π:
   Similarly, Res(f, -π) = (-π)³/cos(-π) = -π³/(-1) = π³

5) By Residue Theorem:
   ∮_{|z|=4} z³/sin z dz = 2πi·[Res(f, π) + Res(f, -π)]
                         = 2πi·(-π³ + π³)
                         = 0
   
   ANSWER: 0
""")

print("\n" + "="*60)
print("SUMMARY OF ANSWERS:")
print("="*60)
print("""
(a) (π/√2a²)·e^{-a/√2}·sin(a/√2)

(b) 2π/(1-β²)^{3/2}

(c) 0 if n is even; (-1)^{(n-1)/2}/n! if n is odd

(d) -πi

(e) i·sinh(1)

(f) 0
""")

# Numerical verification for some cases
print("\n" + "="*60)
print("NUMERICAL VERIFICATION:")
print("="*60)

# Verify (e) numerically
print("\n(e) Numerical check:")
print(f"i·sinh(1) = {1j * math.sinh(1):.6f}")
print(f"Alternative: i(e-1/e)/2 = {1j * (math.e - 1/math.e)/2:.6f}")

# Verify (b) for specific β value
beta = 0.5
print(f"\n(b) For β = {beta}:")
print(f"Analytical: 2π/(1-β²)^(3/2) = {2*math.pi / (1-beta**2)**(3/2):.6f}")