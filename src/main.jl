using Printf
using Statistics
using .LDIPM
using .ProblemGenerator

function DoTest(; num_vars, num_ineqs, rank_W)
    problems = ProblemGenerator.RandomDataSet(30, num_var = num_vars, num_ineq = num_ineqs, rank_W = rank_W, identity_initialization = false, seed = 1)
    
    iters_log = []
    iters_dual = []
    iters_primal = []

    max_iters = 40
    target_duality_gap = 1e-3

    while ProblemGenerator.next(problems)
        P, q, A, u = ProblemGenerator.GetProblemData(problems)
        problem = ProblemGenerator.GetProblemName(problems)

        if ProblemGenerator.InitialGuessAvailable(problems)
            initial_guess = ProblemGenerator.GetInitialGuess(problems)

            algo = LDIPM.longstep(P, q, -A, u)
            
            push!(iters_log, LDIPM.solve(algo, iters = max_iters,  step_type = "log", target_duality_gap = target_duality_gap)) 

            push!(iters_dual, LDIPM.solve(LDIPM.longstep(P, q, -A, u), iters = max_iters,  step_type = "dual", target_duality_gap = target_duality_gap))

            push!(iters_primal, LDIPM.solve(LDIPM.longstep(P, q, -A, u), iters = max_iters, step_type = "primal", target_duality_gap = target_duality_gap))
        end
    end

    println("FINISHED!")

    format_for_paper = false

    mean_iters_log = length(iters_log) > 0 ? mean(iters_log) : NaN
    mean_iters_dual = length(iters_dual) > 0 ? mean(iters_dual) : NaN
    mean_iters_primal = length(iters_primal) > 0 ? mean(iters_primal) : NaN

    if format_for_paper
        string_data = "NumVars: $(num_vars), NumIneqs: $(num_ineqs), RankW: $(rank_W)   "
        string_results = "Log: $(mean_iters_log), Dual: $(mean_iters_dual), Primal: $(mean_iters_primal)   "
    else
        string_data = "$(num_vars) & $(num_ineqs) & $(rank_W) &   "
        string_results = "$(mean_iters_log) & $(mean_iters_dual) & $(mean_iters_primal) \\\\   "
    end
    
    string = string_data * string_results
    
    println(string)
    
end

function DoTestHelper(n)
    DoTest(num_ineqs = 200*n, rank_W = 0*n, num_vars = 100*n)
    DoTest(num_ineqs = 200*n, rank_W = 50*n, num_vars = 100*n)
    DoTest(num_ineqs = 200*n, rank_W = 100*n, num_vars = 100*n)
    DoTest(num_ineqs = 100*n, rank_W = 50*n, num_vars = 100*n)
    DoTest(num_ineqs = 150*n, rank_W = 50*n, num_vars = 100*n)
    DoTest(num_ineqs = 200*n, rank_W = 50*n, num_vars = 100*n)
end

DoTestHelper(1)
#DoTestHelper(10)
