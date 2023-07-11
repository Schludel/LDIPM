module ProblemGenerator

using LinearAlgebra
using Random

function NormalizeRows(M)
    for i in 1:size(M, 1)
        M[i, :] = M[i, :]/norm(M[i, :])
    end
    return M
end

mutable struct RandomDataSet
    num_instances::Int
    num_var::Int
    num_ineq::Int
    rank_W::Int
    seed::Int
    identity_initialization::Bool
    instance_number::Int
    initial_guess_available::Bool
    A
    P
    q
    u
    initial_guess

    RandomDataSet(num_instances; num_var=40, num_ineq=100, rank_W=-1, seed=-1, identity_initialization=false) = new(
        num_instances, num_var, num_ineq, (rank_W == -1 ? max(num_var - num_ineq, 0) : rank_W), seed, identity_initialization, 0, true
    )
end

function next(d::RandomDataSet)
    d.instance_number += 1
    if (d.instance_number > d.num_instances)
        return false
    end
    num_var = d.num_var
    rank_P = d.rank_W
    num_ineq = d.num_ineq

    d.A = NormalizeRows(randn(num_ineq, num_var))

    x = randn(num_var, 1)
    e = ones(num_ineq, 1)

    if (d.identity_initialization)
        s0 = e
        l0 = e
    else
        eps = 0.1
        s0 = e + eps*abs.(randn(num_ineq, 1))
        l0 = e + eps*abs.(randn(num_ineq, 1))
    end

    d.u = s0 + d.A * x

    if rank_P > 0
        Psqrt = NormalizeRows(randn(num_var, rank_P))
        d.P = Psqrt * transpose(Psqrt)
    else
        d.P = zeros(num_var, num_var)
    end

    d.q = -(transpose(d.A) * l0 + d.P * x)

    d.initial_guess = Dict("x" => x, "s" => s0, "l" => l0)
    return true
end

GetProblemData(d::RandomDataSet) = (d.P, d.q, d.A, d.u)

GetProblemName(d::RandomDataSet) = string(d.instance_number)

InitialGuessAvailable(d::RandomDataSet) = d.initial_guess_available

GetInitialGuess(d::RandomDataSet) = d.initial_guess

end