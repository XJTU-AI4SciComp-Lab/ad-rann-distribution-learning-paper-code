function e = relative_l2(pred, ref)
%RELATIVE_L2 Discrete relative L2 error.

    e = norm(pred-ref,2)/max(norm(ref,2),eps);
end
