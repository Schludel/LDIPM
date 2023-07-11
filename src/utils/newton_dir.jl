function newton_dir(l::LDIPM.longstep, r::Matrix{Float64}, v::Matrix{Float64})
    if isempty(l.v_cached) || norm(l.v_cached - v) > 0
        l.v_cached = v
        D = 0 * v * v'
        expv = 0 * v
        for i in 1:l.num_ineq
            D[i, i] = exp(v[i])
            expv[i] = exp(v[i])
        end

        l.DA = D * l.A
        l.Db = D * l.b
        G = l.DA' * l.DA  + l.W
        l.cholesky_factor = cholesky(G)
    end

    x = l.cholesky_factor \ (l.DA' * ( 2 * r) - l.q - l.DA' * l.Db)

    s = l.DA * x + l.Db
    d = r .- s
    d = d ./ r

    beta = 1
    norminf = norm(d, Inf)
    stepsize = 1.0 / max(1.0, 1.0 / (2 * beta) * norminf^2)

    return d, x, l.A * x + l.b, stepsize
end
