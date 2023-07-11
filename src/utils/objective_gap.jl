function gap(l::longstep; v, r)
    n = size(l.A, 1)
    e = ones(n, 1)
    d, x, s, stepsize = newton_dir(l, r, v)
    λ = r .* e .+ d
    s = r .* e .- d
    return float(λ' * s), float(Objective(l.W, l.q, x))
end