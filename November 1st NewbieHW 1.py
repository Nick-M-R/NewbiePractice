# November 1st NewbieHW 1
# Team newbie

# imports
import numpy as np
import matplotlib.pyplot as plt

# Problem 2: Grid Generation & Polar coordinates

# Unit variable setups
R_cyl = 1.0
U_inf = 1

# Grid variable setup
rmin = R_cyl
rmax = 3
N_r = 11
r = np.linspace(rmin,rmax,N_r)

#Jonathan

#Polar Grid continued
Spacing = 41
theta = np.linspace(0, 2 * np.pi, Spacing)
[RR, Theta] = np.meshgrid(r, theta)

XX_p = RR * np.cos(Theta)
YY_p = RR * np.sin(Theta)

#Velocity field Calculation
Gamma_vec = [0, -2, -4, -5]

for i in range(4):
    V_r = U_inf * np.cos(Theta) * (1 - (R_cyl**2 / RR**2))
    V_theta = -U_inf * np.sin(Theta) * (1 + (R_cyl**2/RR**2)) + (Gamma_vec[i] * np.pi) / (2 * np.pi * r)

    V_x = -np.sin(Theta) * V_theta + np.cos(Theta) * V_r
    V_y = np.cos(Theta) * V_theta + np.sin(Theta) * V_r

    V_t = np.sqrt(V_x**2 + V_y**2)

    #Cp Calculation
    Cp = 1 - (V_r**2 + V_theta**2)/(U_inf**2)

    #Convert pressure to rectangular coordinates
    Cp_x = -np.cos(Theta) * Cp
    Cp_y = -np.sin(Theta) * Cp

    Cp_surf = Cp[:, 1]

    #Cl and Cd calculations
    Cl_mat = -0.5 * np.trapz(Cp_surf * np.sin(Theta[:, 1]), theta)
    Cd_mat = -0.5 * np.trapz(Cp_surf * np.cos(Theta[:, 1]), theta)

    #Figure Plotting
    plt.figure()

    levels = 20
    plt.subplot(2, 2, 1)
    plt.contour(XX_p, YY_p, Cp, levels)
    plt.colorbar()
    plt.title("Cp Contour")

    plt.subplot(2, 2, 2)
    plt.contour(XX_p, YY_p, V_r, levels)
    plt.colorbar()
    plt.title("Contour of Radial Velocities")

    plt.subplot(2, 2, 3)
    plt.contour(XX_p, YY_p, V_theta, levels)
    plt.colorbar()
    plt.title("Contour of Tangential Velocities")

    plt.subplot(2, 2, 4)
    plt.contour(XX_p, YY_p, V_t, levels)
    plt.colorbar()
    plt.title("Contour of Total Velocities")

    plt.suptitle(f"Flow Over a Rotating Cylinder, Γ = {Gamma_vec[i]}π, C_L = {round(Cl_mat, 5)}, C_D = {round(Cl_mat, 5)}")

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.quiver(XX_p, YY_p, V_x, V_y)
    plt.title("Velocity Vector Field")

    plt.subplot(2,2,2)
    plt.plot(theta / np.pi, Cp_surf)
    plt.xlabel('Theta (π)')
    plt.ylabel('C_p')
    plt.title("Cp on the Surface")

    plt.subplot(2,2,3)
    plt.quiver(XX_p, YY_p, Cp_x, Cp_y)
    plt.title("Pressure Vector Field")

    plt.subplot(2,2,4)
    plt.plot(theta / np.pi, V_t[:, 1])
    plt.xlabel('Theta (π)')
    plt.ylabel('Total Velocity on Surface')
    plt.title("V_T on the Surface")

    plt.suptitle(f"Flow Over a Rotating Cylinder, Γ = {Gamma_vec[i]}π, C_L = {round(Cl_mat, 5)}, C_D = {round(Cd_mat, 5)}")

plt.show()

# Problem 3

# Gamma setup
spaces = 101
Gamma = np.linspace(0, -1*np.pi, spaces) #******************** Ask Nick about >1 arcsin calculation when -5

# Numerically
for t in range(spaces):
    #Setup of another velocity calculation
    V_r2 = U_inf*np.cos(Theta)*(1 - (R_cyl**2/RR**2))
    V_theta2 = - U_inf*np.sin(Theta)*(1 + (R_cyl**2/RR**2)) + (Gamma[t]*np.pi)/(2*np.pi*r)
    #convert to rectangular coordinates 
    V_x2 = - np.sin(Theta)*V_theta2 + np.cos(Theta)*V_r2
    V_y2 = np.cos(Theta)*V_theta2 + np.sin(Theta)*V_r2
    #find total velocity
    V_t2 = (V_x2**2+V_y2**2)**(1/2)
    #Cp calculation
    Cp2= 1 - (V_r2**2 + V_theta2**2)/(U_inf**2)
    Cp_surf2 = Cp2[:,1]
    #Scott 
    #Numerically
    Cl_mat2 = (-1/2 * np.trapz((Theta[:,1]), (Cp_surf2*np.sin(Theta[:,1]))))/np.pi
    #Analytically
    Cl_ana2 = -(Gamma[t])/(U_inf*R_cyl)

    #Stagnation Calculations

    #Indexing
    #min = np.argmin(np.abs(V_t2[11:30,1]))
    array_1 = np.abs(V_t2[11:30,1])
    index1 = np.where(array_1 == array_1.min())

    #[_,index1] = min(abs(V_t2[11:30,1]))

    #Mirror left and right sides
    stag_num_right = (Theta[index1,1] - 3/2 * np.pi)/np.pi
    stag_num_left = (-stag_num_right-1)
    stag_ana = (np.arcsin((Gamma[t])/(4*np.pi*U_inf*R_cyl)))/np.pi
#plot results
plt.figure()

plt.subplot(1,2,1)
plt.plot(Gamma, Cl_mat2, 'b-')
plt.plot(Gamma, Cl_ana2, 'r')
plt.xlabel('Circulation Gamma')
plt.ylabel('C_1')
plt.title('C_1 vs Circulation')
plt.legend('Numerical','Analytical')

plt.subplot(1,2,2)
plt.plot(Gamma*np.pi, stag_num_right, 'b-')
plt.xlim([min(Gamma)],max(Gamma*np.pi))
plt.plot(Gamma*np.pi,stag_num_left, color = 'green',linestyle ='solid')
plt.xlim([min(Gamma),max(Gamma*np.pi)])
plt.plot(Gamma, stag_ana, 'r-')
plt.xlabel('Circulation Gamma')
plt.ylabel('Theta_s_t_a_g (pi)')
plt.title('Stagnation Angle vs Circulation')
plt.legend('Numerical Right','Numerical left','Analytical Right')




