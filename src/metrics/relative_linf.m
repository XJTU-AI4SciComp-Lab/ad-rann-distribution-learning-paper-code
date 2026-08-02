function e = relative_linf(pred, ref)
%RELATIVE_LINF Discrete relative Linf error.

    e = norm(pred-ref,inf)/max(norm(ref,inf),eps);
end
