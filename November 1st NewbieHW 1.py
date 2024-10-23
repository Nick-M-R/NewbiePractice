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

Spacing = 41
interval = 2/(Spacing -1)

theta = linspace(0,2*pi, Spacing)
[RR, Theta] = meshgrid(r,theta)

#Polar grid into cartesian form
XX_p = RR.*cos(Theta)
YY_p = RR.*sin(Theta)

# Velocity field Calculation
n = 0

# Gamma used in calculations
Gamma_vec = [0, -2, -4, -5]

for i == 1:4:
    n += 1

V_r = U_inf*cos(Theta).*(1 - (R_cyl^2./RR.^2))
V_theta = U_inf*sin(Theta).*(1 + (R__cyl^2./RR.^2)) + (Gamma_vec(n)*pi)./(2.*pi.*r)

#convert back to cartesian for plotting
V_x = -sin(Theta).*V_theta + cos(Theta).*V_r
V_y = cos(Theta).*V_theta + sin(Theta).*V_r

V_t = (V_x.^2 + V_y^2).^(1/2)

#Cp Calculation
Cp = 1 - (V_r.^2+V_theta.^2)/(U_inf^2)

#convert pressure to rectangular coordinates
Cp_x = -cos(Theta).*Cp
Cp_y = -sin(Theta).*Cp

Cp_surf = Cp(:,1)

#Cl and Cd calculations

#Numerically
Cl_mat = -1/2 * trapz((Theta(:,1)), (Cp_surf.*sin(Theta(:,1))))
Cd_mat = -1/2 * trapz(Theta(:,1), Cp_surf.*cos(Theta(:,1)))

#Analytically 

#Multiply directional Cp by the area and sum along surface
Cl_ana = sum(Cp_y(:,1))*(interval*pi/2)
Cd_ana = sum(Cp_x(:,1))*(interval*pi/2)

figure()

subplot(2,2,1)
contourf(XX_p, YY_p, Cp, levels) #contour plot filled with color
colorbar
title("Cp Contour")

subplot (2,2,2)
contourf(XX_p, YY_p, V_r, levels) #coutour plot filled with color
colorbar
title("Contour of Radial Velocities")

subplot (2,2,3)
contourf(XX_p, YY_p, V_theta, levels) #coutour plot filled with color
colorbar
title("Contour of Tangential Velocities")

subplot (2,2,4)
contourf(XX_p, YY_p, v_t, levels) #coutour plot filled with color
colorbar
title("Contour of Total Velocities")

sgtitle("Flow Over a Rotating Cylinder"+", \Gamma ="+Gamma_vec(n)+"\pi"+", C_L ="+round(Cl_mat,5)+", C_D ="+round(Cd_mat,5))

figure()
subplot(2,2,1)
quiver(XX_p, YY_p, V_x, V_y) #change N_r = 5, N_theta = 21 for better arrow plotting
axis equal
title("Velocity Vector Field")

subplot(2,2,2)
plot((Theta./pi),Cp_surf)
xlabel('Theta(\pi)')
ylabel('C_p')
title("Cp on the surface")

subplot(2,2,3)
quiver(XX_p, YY_p, Cp_x, Cp_y)
axis equal
title("Pressure Vector Field")

subplot(2,2,4)
plot((Theta./pi), V_t(:,1))
xlabel('Theta (\pi)')
ylabel('Total Velocity on Surface')
title("V_T on the surface")

sgtitle("Flow over a rotating cylinder"+", \Gamma ="+Gamma_vec(n)+"\pi"+", C_L ="+round(Cl_mat,5)+", C_D ="+round(Cd_mat,5))

end