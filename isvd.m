

function [Q,S,R] = InitializeISVD(u1,W)

    S = (u1'* W *u1)^(1/2); % Num
    Q = u1 * S^(-1);        % m*1
    R = eye(1);
end

function [q, V, Q_0, Q,S,R] = UpdateISVD3(q, V, Q_0, Q, S, R, u_l_new, W, tol)

    d = Q' * (W * u_l_new);   
    e = u_l_new - Q*d; p = sqrt(e'* W *e);   
    if p < tol
        q = q + 1;
        V{q} = Q_0' * d;
    else
        if q > 0
            k = size(S,1);
            Y = [S, cell2mat(V)]; 
            [Qy, Sy, Ry] = svd(Y, 'econ');
            Q = Q*(Q_0*Qy); S = Sy;
            R1 = Ry(1:k,:); R2 = Ry(k+1:end,:); 
            R = [R*R1; R2];
            d = Qy' * d;
        end
        V = {}; q = 0;
        
        e = e / p;

        if sqrt(e' * W * Q(:,1)) > tol
            e = e - Q * (Q' * (W*e)); 
            p1 = (e'* W *e)^(1/2); e = e / p1;
        end
        
        k = size(S,1);
        Y = [S, Q_0' * d; zeros(1, k), p];
        [Qy, Sy, Ry] = svd(Y);
        Q_0 = [Q_0, zeros(size(Q_0, 1), 1); zeros(1, size(Q_0, 2)), 1] * Qy;
        Q = [Q, e]; S = Sy;
        R = [R, zeros(size(R, 1), 1); zeros(1, size(R, 2)), 1] * Ry;         
    end
end


function [Q, S, R] = UpdateISVD3check(q,V,Q_0,Q,S,R)

    k = size(S, 1);
    if q > 0
        Y = [S, cell2mat(V)];
        [Qy, Sy, Ry] = svd(Y, 'econ');
        
        Q = Q*(Q_0*Qy); S = Sy; 
        R1 = Ry(1:k,:); R2 = Ry(k+1:end,:); 
        R = [R*R1; R2];
    else
        Q = Q * Q_0;
    end
end

clc, clear
tol = 1e-15;

U = rand(100,30)*rand(30,80); 
m = size(U,1); W = eye(m); u1 = U(:,1);


tic
[Q, S, R] = InitializeISVD(u1, W);
V = {}; Q_0 = 1; q = 0;
n = size(U,2);
for L = 2:n
    u_l_new = U(:,L);
    [q, V, Q_0, Q, S, R] = UpdateISVD3(q,V,Q_0,Q,S,R,u_l_new,W,tol);
end
[Q, S, R] = UpdateISVD3check(q,V,Q_0,Q,S,R);
toc


tic
[Q_st,S_st,R_st] = svd(U); 
toc

norm(Q*S*R' - U)
norm(Q_st*S_st*R_st' - U)
norm(abs(Q(:,end)' * W * Q(:,1)))


