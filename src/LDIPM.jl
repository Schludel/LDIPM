module LDIPM 

using LinearAlgebra
using Random
using SparseArrays
using SuiteSparse

function Objective(P, q, x)
    x = vec(x)
    q = vec(q)
    val = .5 * (x' * P * x) + q' * x
    return val
end

mutable struct longstep
    q
    A
    b
    G
    W
    num_ineq::Int
    e
    v_cached
    DA
    Db
    cholesky_factor
    up_low

    longstep(W, q, A, b) = new(
        reshape(q, :, 1), A, reshape(b, :, 1), A' * A  + W, W, size(A, 1), ones(size(A, 1), 1), [], [], [], nothing, nothing
    )
end

include("utils/newton_dir.jl")
include("utils/line_search.jl")
include("utils/newton_step.jl")
include("utils/objective_gap.jl")

log_func(l::longstep, x) = log.(max.(x, 1e-15))

function solve(l::longstep; fixed_point_iters = 1, newton_iters = 1, iters = 10, step_type = "log", target_duality_gap = 0)
    num_ineqs = size(l.A, 1)
    progress_last = Inf
    v = zeros(num_ineqs, 1)
    r = ones(num_ineqs, 1)
    verbose = false
    dinf_bound = 0.99

    if verbose
        println("Step-type: ", step_type)
    end

    for iter_cnt in 1:iters
        success, alpha_min, alpha_max = line_search(l, r = ones(num_ineqs, 1), v = v, dinf_bound = dinf_bound)
        if success
            r = ones(num_ineqs, 1) ./ abs(alpha_max)
        else
            println(iter_cnt)
            println(alpha_min)
            println(alpha_max)
            println("line search failed")
            error()
        end

        final_gap = gap(l, r=r, v=v)
        d, x, s, stepsize = newton_dir(l, r, v)
        dinf = norm(d, Inf)
        if abs(dinf - dinf_bound) > 1e-2
            println("line search failed")
            error()
        end

        mu = float((r' * r) / num_ineqs)
        if dinf <= 1.0 && mu[1] < target_duality_gap
            return iter_cnt
        end

        v, damping = Newton(l, r, v, newton_iters, 1e10, step_type)
        v_last = v
        format_str = "{:.2e}"
        format_str_int = "{:3d}"
        if verbose
            string = ""
            string = string * " gap " * format(format_str, final_gap[1])
            string = string * " obj " * format(format_str, final_gap[2])
            string = string * " dinf" * format(format_str, dinf)

            println(string)
        end
    end
end

end
