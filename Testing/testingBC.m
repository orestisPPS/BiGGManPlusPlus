% Ellipsoid parameters
a = 4; % semi-axis in the x-direction
b = 3; % semi-axis in the y-direction
c = 2; % semi-axis in the z-direction

% Generate u, v parameters for creating ellipsoid
[u,v] = meshgrid(linspace(0, 2*pi, 50), linspace(-pi/2, pi/2, 50));

% Generate ellipsoid points using parametric equations
x = a*cos(u) .* cos(v);
y = b*cos(u) .* sin(v);
z = c*sin(u);

% Plot the ellipse
figure
surf(x, y, z)

% Adjust plot appearance
axis equal
xlabel('X')
ylabel('Y')
zlabel('Z')
title('3D Ellipse')

