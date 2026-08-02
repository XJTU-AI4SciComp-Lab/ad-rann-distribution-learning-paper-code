function D = gaussian_derivatives(X, p, basis)
%GAUSSIAN_DERIVATIVES Gaussian values and analytic p-derivatives.

    p = p(:);

    Z1 = basis.Z(1,:);
    Z2 = basis.Z(2,:);

    C1 = basis.C(1,:);
    C2 = basis.C(2,:);

    W1 = p(1)*Z1;
    W2 = p(2)*Z2;

    Q1 = (X(:,1)-C1).*Z1;
    Q2 = (X(:,2)-C2).*Z2;

    S = p(1)*Q1 + p(2)*Q2;

    E = exp(-S.^2);
    F1 = -2*S.*E;
    H = (4*S.^2-2).*E;
    Hp = (12*S-8*S.^3).*E;

    D.phi = E;
    D.xx = H.*(W1.^2);
    D.yy = H.*(W2.^2);

    D.dp = cell(2,1);

    D.dp{1}.phi = F1.*Q1;
    D.dp{2}.phi = F1.*Q2;

    D.dp{1}.xx = H.*(2*W1.*Z1) + Hp.*Q1.*(W1.^2);
    D.dp{1}.yy = Hp.*Q1.*(W2.^2);

    D.dp{2}.xx = Hp.*Q2.*(W1.^2);
    D.dp{2}.yy = H.*(2*W2.*Z2) + Hp.*Q2.*(W2.^2);
end
