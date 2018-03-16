%Nearest Neighbor (NN) Algorithm - Performs NN classification on digits
%from a subset containing 5000 training images and 500 test images of the
%MNIST dataset.
%
% Inputs
%
%    none
%
% Outputs: Plots of the following
%    
%    Gradient descent loss function as a function of the iterations
%    Gradient descent clock bias error as a function of the iterations
%    Gradient descent position error as a function of the iterations
%    Gradient descent stepsize as a function of the iterations
%
%    Gauss-Newton loss function as a function of the iterations
%    Gauss-Newton clock bias error as a function of the iterations
%    Gauss-Newton position error as a function of the iterations
%    Gauss-Newton stepsize as a function of the iterations
%
% Other m-files required: none
% Subfunctions: none
%
% Author: Erik Sandström, Lund University
% email address: erik.sandstrm@gmail.com
% Website: http://dsp.ucsd.edu/~kreutz/ECE174.html
% March 2018

%------------- BEGIN CODE --------------
clear all
close all
%Conversion factor from Earth Radians to meters
convToM = 6370*10^3;
%Actual position of vehicle
S = [1 0 0]';
%Actual clock bias error
b = 2.354788068*10^(-3);
%Actual position of satellites
S1 = [3.5852 2.0700 0]';
S2 = [2.9274 2.9274 0]';
S3 = [2.6612 0 3.1712]';
S4 = [1.4159 0 3.8904]';
S_tot = [S1 S2 S3 S4]';
%Initial vehicle estimate
S_hat_0 = [0.9331 0.25 0.258819]';
b_hat_0 = 0;
X_hat_k_grad = [S_hat_0 ; b_hat_0];
X_hat_k_gauss = [S_hat_0 ; b_hat_0];
%Termination criteria
maxItr = 10^6;
deltaXMin = 10^(-8);
%Stepsize parameter
alphak_gradVec = [0.1 0.2 0.3 0.3125];
alphak_gaussVec = [0.7 1 1.3 1.8];
%Matrices to save data for different values of alphak_grad and alphak_gauss
Grad_Loss_Matrix = 0;
Grad_Clock_Matrix = 0;
Grad_Pos_Matrix = 0;
Grad_DeltaX_Matrix = 0;

Gauss_Loss_Matrix = 0;
Gauss_Clock_Matrix = 0;
Gauss_Pos_Matrix = 0;
Gauss_DeltaX_Matrix = 0;

%Generate synthetic data which otherwise would be
%measured by the phone.
y = [((S-S1)'*(S-S1))^(1/2)+b ((S-S2)'*(S-S2))^(1/2)+b ...
    ((S-S3)'*(S-S3))^(1/2)+b ((S-S4)'*(S-S4))^(1/2)+b]';


%% Gradient Descent Algorithm

%Run for different values of alphak_grad
legendInfofig1 = {};
legendInfofig2 = {};
legendInfofig3 = {};
legendInfofig4 = {};

%Iterate through all four values of alpha_k
for p = 1:4
alphak_grad = alphak_gradVec(p);
%Reset estimate to initial condition
X_hat_k_grad = [S_hat_0 ; b_hat_0];

%Calculate Jacobian transpose
H_X_hat_k = zeros(4);
for i = 1:4
    for j = 1:3
        H_X_hat_k(i,j) = (X_hat_k_grad(j)-S_tot(i,j))/...
            ((X_hat_k_grad(1:3)-S_tot(i,:)')'*...
            (X_hat_k_grad(1:3)-S_tot(i,:)'))^(1/2); 
    end
end
H_X_hat_k(:,4) = ones(4,1);

%Calculate small h
h_X_hat_k = zeros(4,1);
for i = 1:4
h_X_hat_k(i) = ((X_hat_k_grad(1:3)-S_tot(i,:)')'*...
    (X_hat_k_grad(1:3)-S_tot(i,:)'))^(1/2) + X_hat_k_grad(4);
end
h_X_Grad = h_X_hat_k;

deltaX_k = alphak_grad*H_X_hat_k'*(y-h_X_hat_k); %First value of the step size
XvecGrad = X_hat_k_grad; %Matrix to store all values of X
deltaXGrad = deltaX_k; %Matrix to store all values of deltaX
normDeltaX_k = (deltaX_k'*deltaX_k)^(1/2);
counter = 1;
while(normDeltaX_k > deltaXMin && counter < maxItr)
    X_hat_k_grad = X_hat_k_grad + deltaX_k;
    XvecGrad = [XvecGrad X_hat_k_grad]; %Add the newly calculated estimate to XvecGrad
    counter = counter + 1;
    
    %Calculate the Jacobian matrix
    for i = 1:4
        for j = 1:3
            H_X_hat_k(i,j) = (X_hat_k_grad(j)-S_tot(i,j))/...
                ((X_hat_k_grad(1:3)-S_tot(i,:)')'*...
                (X_hat_k_grad(1:3)-S_tot(i,:)'))^(1/2); 
        end
    end
    H_X_hat_k(:,4) = ones(4,1);
    %Calculate small h
    h_X_hat_k = zeros(4,1);
    for i = 1:4
        h_X_hat_k(i) = ((X_hat_k_grad(1:3)-S_tot(i,:)')'*...
            (X_hat_k_grad(1:3)-S_tot(i,:)'))^(1/2) + X_hat_k_grad(4);
    end
    h_X_Grad = [h_X_Grad h_X_hat_k]; %Matrix which stores all values of
    %h(X) which is used when plotting the loss function
    
    deltaX_k = alphak_grad*H_X_hat_k'*(y-h_X_hat_k);
    deltaXGrad = [deltaXGrad deltaX_k];
    normDeltaX_k = (deltaX_k'*deltaX_k)^(1/2); 
end

%Plot the loss function as a function of the iterations
figure(1)
hold on
step1LossGrad = [y(1)*ones(1,counter) ; y(2)*ones(1,counter)...
    ; y(3)*ones(1,counter) ; y(4)*ones(1,counter)] - h_X_Grad;
step2LossGrad = step1LossGrad.^2;
step3LossGrad = step2LossGrad(1,:)+step2LossGrad(2,:)+...
    step2LossGrad(3,:)+step2LossGrad(4,:);
LossGrad = 1/2*step3LossGrad;
semilogy(0:length(LossGrad)-1,LossGrad);
legendInfofig1{p} = ['\alpha_k = ' num2str(alphak_grad)];
hold off

%Plot the clock bias error as a function of the iterations
figure(2)
hold on
DiffXGrad = [ones(1,counter) ; zeros(2,counter) ; b*ones(1,counter)] - XvecGrad;
semilogy(0:length(DiffXGrad(4,:))-1, convToM*abs(DiffXGrad(4,:)));
legendInfofig2{p} = ['\alpha_k = ' num2str(alphak_grad)];
hold off

%Plot the error in position as a function of the iterations
figure(3)
hold on
ErrSquared = DiffXGrad(1:3,:).^2;
ErrPos = convToM*(ErrSquared(1,:)+ErrSquared(2,:)+ErrSquared(3,:)).^(1/2);
semilogy(0:length(ErrPos)-1, ErrPos);
legendInfofig3{p} = ['\alpha_k = ' num2str(alphak_grad)];
hold off

%Plot convergence rate of stepsize as a function of the iterations
figure(4)
hold on
deltaXGradSquared = deltaXGrad.^2;
normDeltaXGrad = (deltaXGradSquared(1,:)+deltaXGradSquared(2,:)...
    +deltaXGradSquared(3,:)+deltaXGradSquared(4,:)).^(1/2);
semilogy(0:length(normDeltaXGrad)-1, normDeltaXGrad);
legendInfofig4{p} = ['\alpha_k = ' num2str(alphak_grad)];
hold off
end

figure(1)
title('Gradient descent loss function as a function of the iterations');
xlabel('Iteration step k');
ylabel('Loss function L_k');
legend(legendInfofig1);
set(gca,'yscale','log');

figure(2)
title('Gradient descent clock bias error b as a function of the iterations');
xlabel('Iteration step k');
ylabel('Clock bias b_k (m)');
legend(legendInfofig2);
set(gca,'yscale','log');

figure(3)
title('Gradient descent position error as a function of the iterations');
xlabel('Iteration step k');
ylabel('Position error ||S-S_k|| (m)');
legend(legendInfofig3);
set(gca,'yscale','log');

figure(4)
title('Gradient descent stepsize \DeltaX_k as a function of the iterations');
xlabel('Iteration step k');
ylabel('Stepsize \DeltaX_k');
legend(legendInfofig4);
set(gca,'yscale','log');
%% Gauss-Newton Algorithm

%Run for different values of alphak_gauss
legendInfofig5 = {};
legendInfofig6 = {};
legendInfofig7 = {};
legendInfofig8 = {};

%Iterate through all four values of alpha_k
for p = 1:4
alphak_gauss = alphak_gaussVec(p);
%Reset estimate to initial condition
X_hat_k_gauss = [S_hat_0 ; b_hat_0];

%Calculate Jacobian transpose
H_X_hat_k = zeros(4);
for i = 1:4
    for j = 1:3
        H_X_hat_k(i,j) = (X_hat_k_gauss(j)-S_tot(i,j))/...
            ((X_hat_k_gauss(1:3)-S_tot(i,:)')'*...
            (X_hat_k_gauss(1:3)-S_tot(i,:)'))^(1/2); 
    end
end
H_X_hat_k(:,4) = ones(4,1);

%Calculate small h
h_X_hat_k = zeros(4,1);
for i = 1:4
h_X_hat_k(i) = ((X_hat_k_gauss(1:3)-S_tot(i,:)')'*...
    (X_hat_k_gauss(1:3)-S_tot(i,:)'))^(1/2) + X_hat_k_gauss(4);
end
h_X_Gauss = h_X_hat_k;

deltaX_k = alphak_gauss*H_X_hat_k\(y-h_X_hat_k); %First value of the
%step size
XvecGauss = X_hat_k_gauss; %Matrix to store all values of X
deltaXGauss = deltaX_k; %Matrix to store all values of deltaX
normDeltaX_k = (deltaX_k'*deltaX_k)^(1/2);  
counter = 1;
while(normDeltaX_k > deltaXMin && counter < maxItr)
    X_hat_k_gauss = X_hat_k_gauss + deltaX_k;
    XvecGauss = [XvecGauss X_hat_k_gauss]; %Add the newly 
    %calculated estimate to XvecGauss
    counter = counter + 1;
    
     %Calculate the Jacobian matrix
    for i = 1:4
        for j = 1:3
            H_X_hat_k(i,j) = (X_hat_k_gauss(j)-S_tot(i,j))/...
                ((X_hat_k_gauss(1:3)-S_tot(i,:)')'*...
                (X_hat_k_gauss(1:3)-S_tot(i,:)'))^(1/2); 
        end
    end
    H_X_hat_k(:,4) = ones(4,1);
    
    %Calculate small h
    h_X_hat_k = zeros(4,1);
    for i = 1:4
        h_X_hat_k(i) = ((X_hat_k_gauss(1:3)-S_tot(i,:)')'*...
            (X_hat_k_gauss(1:3)-S_tot(i,:)'))^(1/2) + X_hat_k_gauss(4);
    end
    h_X_Gauss = [h_X_Gauss h_X_hat_k]; %Matrix which stores all values of
    %h(X) which is used when plotting the loss function

    deltaX_k = alphak_gauss*H_X_hat_k\(y-h_X_hat_k);
    deltaXGauss = [deltaXGauss deltaX_k];
    normDeltaX_k = (deltaX_k'*deltaX_k)^(1/2);
end

%Plot the loss function as a function of the iterations
figure(5)
step1LossGauss = [y(1)*ones(1,counter) ; y(2)*ones(1,counter) ;...
    y(3)*ones(1,counter) ; y(4)*ones(1,counter)] - h_X_Gauss;
step2LossGauss = step1LossGauss.^2;
step3LossGauss = step2LossGauss(1,:)+step2LossGauss(2,:)+...
    step2LossGauss(3,:)+step2LossGauss(4,:);
LossGauss = 1/2*step3LossGauss;
hold on
semilogy(0:length(LossGauss)-1,LossGauss);
legendInfofig5{p} = ['\alpha_k = ' num2str(alphak_gauss)];
hold off

%Plot the clock bias error as a function of the iterations
figure(6)
hold on
DiffXGauss = [ones(1,counter) ; zeros(2,counter) ; b*ones(1,counter)] - XvecGauss;
semilogy(0:length(DiffXGauss(4,:))-1, convToM*abs(DiffXGauss(4,:)));
legendInfofig6{p} = ['\alpha_k = ' num2str(alphak_gauss)];
hold off

%Plot the error in position as a function of the iterations
figure(7)
hold on
ErrSquared = DiffXGauss(1:3,:).^2;
ErrPos = convToM*(ErrSquared(1,:)+ErrSquared(2,:)+ErrSquared(3,:)).^(1/2);
semilogy(0:length(ErrPos)-1, ErrPos);
legendInfofig7{p} = ['\alpha_k = ' num2str(alphak_gauss)];
hold off

%Plot convergence rate of stepsize as a function of the iterations
figure(8)
hold on
deltaXGaussSquared = deltaXGauss.^2;
normDeltaXGauss = (deltaXGaussSquared(1,:)+deltaXGaussSquared(2,:)+...
    deltaXGaussSquared(3,:)+deltaXGaussSquared(4,:)).^(1/2);
semilogy(0:length(normDeltaXGauss)-1, normDeltaXGauss);
legendInfofig8{p} = ['\alpha_k = ' num2str(alphak_gauss)];
hold off
end
figure(5)
title('Gauss-Newton loss function as a function of the iterations');
xlabel('Iteration step k');
ylabel('Loss function L_k');
legend(legendInfofig5);
set(gca,'yscale','log');

figure(6)
title('Gauss-Newton clock bias error b as a function of the iterations');
xlabel('Iteration step k');
ylabel('Clock bias b_k (m)');
legend(legendInfofig6);
set(gca,'yscale','log');

figure(7)
title('Gauss-Newton position error as a function of the iterations');
xlabel('Iteration step k');
ylabel('Position error ||S-S_k|| (m)');
legend(legendInfofig7);
set(gca,'yscale','log');

figure(8)
title('Gauss-Newton stepsize \DeltaX_k as a function of the iterations');
xlabel('Iteration step k');
ylabel('Stepsize \DeltaX_k');
legend(legendInfofig8);
set(gca,'yscale','log');
%------------- END OF CODE --------------
