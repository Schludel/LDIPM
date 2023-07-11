function Newton(l::longstep, r, v0, iters = 30, eps = 1e-3, step_type = "log")
    v = v0
    d = zeros(size(v))  # Initialize `d` to a vector of zeros of the same size as `v`
    stepsize = 0.0  # Initialize `stepsize` to 0.0
    for i in 1:iters
        d, x, s, stepsize = newton_dir(l, r, v)
        if step_type == "dual"
            v = log_func(l, exp.(v) .* (1 .+ stepsize .* d))
        elseif step_type == "primal"
            v = -log_func(l, exp.(-v) .* (1 .- stepsize .* d))
        elseif step_type == "log"
            v = v .+ d .* stepsize
        else
            error("Invalid step-type")
        end

        if norm(d) < eps
            break
        end
    end

    if norm(d) > eps
        println("FAIL")
        println(norm(d))
        error()
    end

    return v, stepsize
end