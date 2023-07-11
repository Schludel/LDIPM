function line_search(l::longstep; r, v, dinf_bound = 1)
    d0 = newton_dir(l, r*1e15, v)[1]
    d1 = newton_dir(l, r, v)[1] - d0
    min_alpha = 0
    max_alpha = 1e15
    for i in 1:l.num_ineq
        temp_min = float((dinf_bound-d0[i]) / d1[i])
        temp_max = float((-dinf_bound-d0[i]) / d1[i])
        if d1[i] > 0
            temp = temp_max
            temp_max = temp_min
            temp_min = temp
        end

        if temp_min > min_alpha
            min_alpha = temp_min
        end

        if temp_max < max_alpha
            max_alpha = temp_max
        end
    end

    return min_alpha < max_alpha, min_alpha, max_alpha
end