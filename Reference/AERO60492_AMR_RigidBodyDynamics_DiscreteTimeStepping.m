clear all;
close all;


%Setup the model parameters
mass = 10;
Ixx = 1;
Iyy = 1;
Izz = 1;
dt = 0.001;              %time step duration
simDuration = 10;       %total sim time in seconds


%Initial states (k=0)
%variables without subscript are assumed to be time-step k
U = 0;
V = 0;
W = 0;
p = 0;
q = 0;
r = 0;
x = 0;
y = 0;
z = 0;
phi = 0;
theta = 0;
psi = 0;


%Setup a doublet input for force Z and torque L
Zin = zeros(length(0:dt:simDuration),1);
Zin(1/dt:2/dt) = 1;
Zin(2/dt:3/dt) = -1;


Lin = zeros(length(0:dt:simDuration),1);
Lin(4/dt:5/dt) = -0.5;
Lin(5/dt:6/dt) = 0.5;




%Iterative loop
simTime = 0;
iterationNo = 1;
state(iterationNo,:) = [U,V,W,p,q,r,x,y,z,phi,theta,psi];
times(iterationNo) = 0;
disp('Sim. starting');
for k=0:dt:simDuration-dt
    simTime = simTime+dt;
    iterationNo = iterationNo+1;

    %Find forces and torques
    %[X,Y,Z] = functionToEvalForces()
    %[L,M,N] = functionToEvalTorques()
    [X,Y,Z] = deal(0,0,0);
    [L,M,N] = deal(0,0,0);

    Z = Zin(iterationNo,1); 
    L = Lin(iterationNo,1); 


    %Evaluate derivatives
    Udot = X/mass + r*V - q*W;
    Vdot = Y/mass + p*W - r*U;
    Wdot = Z/mass + q*U - p*V;

    pdot = (1/Ixx)*(L+(Iyy-Izz)*q*r);
    qdot = (1/Iyy)*(L+(Izz-Ixx)*r*p);
    rdot = (1/Izz)*(L+(Ixx-Iyy)*p*q);


    %Approximate integral/derivative
    Uk1 = U + dt*Udot;
    Vk1 = V + dt*Vdot;
    Wk1 = W + dt*Wdot;

    pk1 = p + dt*pdot;
    qk1 = q + dt*qdot;
    rk1 = r + dt*rdot;


    %Transform into reference axes
    xdot = cos(psi)*cos(theta)*U + (cos(psi)*sin(theta)*sin(phi)-sin(psi)*cos(phi))*V + (cos(psi)*sin(theta)*cos(phi)+sin(psi)*sin(phi))*W;
    ydot = sin(psi)*cos(theta)*U + (sin(psi)*sin(theta)*sin(phi)+cos(psi)*cos(phi))*V + (sin(psi)*sin(theta)*cos(phi)-cos(psi)*sin(phi))*W;
    zdot =       -1*sin(theta)*U +                              cos(theta)*sin(phi)*V +                              cos(theta)*cos(phi)*W;

    phidot   = 1*p + sin(phi)*tan(theta)*q + cos(phi)*tan(theta)*r;
    thetadot = 0*p +            cos(phi)*q -            sin(phi)*r;
    psidot   = 0*p + sin(phi)*sec(theta)*q + cos(phi)*sec(theta)*r;

    
    %Approximate integral/derivative
    xk1 = x + dt*xdot;
    yk1 = y + dt*ydot;
    zk1 = z + dt*zdot;

    phik1   = phi   + dt*phidot;
    thetak1 = theta + dt*thetadot;
    psik1   = psi   + dt*psidot;

    %Update new state as previous state to time-step forward by one
    %iteration
    U = Uk1;
    V = Vk1;
    W = Wk1;
    p = pk1;
    q = qk1;
    r = rk1;
    x = xk1;
    y = yk1;
    z = zk1;
    phi   = phik1;
    theta = thetak1;
    psi   = psik1;

    %Save the state to a time-history
    state(iterationNo,:) = [U,V,W,p,q,r,x,y,z,phi,theta,psi];
    times(iterationNo) = simTime;

end

disp('Sim. complete');

%Plot the outputs
figure('name','inputs');hold on;
plot(times,Zin);
plot(times,Lin);
legend('Z','L');


figure('name','rates');hold on;
plot(times,state(:,1));
plot(times,state(:,2));
plot(times,state(:,3));
plot(times,state(:,4));
plot(times,state(:,5));
plot(times,state(:,6));
legend('U','V','W','p','q','r');

figure('name','pose');hold on;
plot(times,state(:,7));
plot(times,state(:,8));
plot(times,state(:,9));
plot(times,state(:,10));
plot(times,state(:,11));
plot(times,state(:,12));
legend('x','y','z','phi','theta','psi');

