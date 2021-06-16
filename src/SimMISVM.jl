import MLJBase
using LinearAlgebra: norm, I, pinv, Bidiagonal, diag, diagm, svd, svdvals
using Flux: onehotbatch, onecold
using MLJ: levels, nrows
using CategoricalArrays: CategoricalArray

mutable struct SimMISVMClassifier <: MLJBase.Deterministic
    C::Float64
    α::Float64
    β::Float64
    μ::Float64
    ρ::Float64
    maxiter::Int64
    tol::Float64
end

function SimMISVMClassifier(; C=1.0, α=1.0, β=1.0, μ=.1, ρ=1.2, maxiter=300, tol=1e-3)
    @assert all(i -> (i > 0), [C, α, β, μ, maxiter, tol])
    @assert ρ > 1.0
    model = SimMISVMClassifier(C, α, β, μ, ρ, maxiter, tol)
end

function MLJBase.fit(model::SimMISVMClassifier, verbosity::Integer, X, y)
    fitresult = X, y, verbosity
    cache = nothing
    report = nothing

    return fitresult, cache, report
end

function MLJBase.predict(model::SimMISVMClassifier, fitresult, Xnew)
    X, y, verbosity = fitresult
    v = init_vars(model, X, Xnew, y)

    if verbosity > 5
        E_res, F_res, Q_res, T_res, R_res, U_res, Z_res = calc_residuals(model, v)
        res = sum([norm(r) for r in (E_res, F_res, Q_res, T_res, R_res, U_res, Z_res)])

        ol = obj_loss(model, v)
        ll = lagrangian_loss(model, v)
        print("Loss: " * string(ol) * "     \t") 
        print("Lagrangian: " * string(ll) * "     \t") 
        println("Residual: " * string(res))
    end

    for i in 1:model.maxiter
        X_update!(model, v)
        S_update!(model, v)
        W_update!(model, v)
        b_update!(model, v)
        E_update!(model, v)
        F_update!(model, v)
        Q_update!(model, v)
        R_update!(model, v)
        T_update!(model, v)
        U_update!(model, v)

        """
        print("X: "); @time X_update!(model, v)
        print("S: "); @time S_update!(model, v)
        print("W: "); @time W_update!(model, v)
        print("b: "); @time b_update!(model, v)
        print("E: "); @time E_update!(model, v)
        print("F: "); @time F_update!(model, v)
        print("Q: "); @time Q_update!(model, v)
        print("R: "); @time R_update!(model, v)
        print("T: "); @time T_update!(model, v)
        print("U: "); @time U_update!(model, v)
        """

        E_res, F_res, Q_res, T_res, R_res, U_res, Z_res = calc_residuals(model, v)

        v.Λ = v.Λ + v.μ * E_res
        v.Π = v.Π + v.μ * F_res
        v.Σ = v.Σ + v.μ * Q_res
        v.Θ = v.Θ + v.μ * T_res 
        v.Ω = v.Ω + v.μ * R_res
        v.Ξ = v.Ξ + v.μ * U_res
        v.Δ = v.Δ + v.μ * Z_res

        res = sum([norm(r) for r in (E_res, F_res, Q_res, T_res, R_res, U_res, Z_res)])

        if verbosity > 5
            ol = obj_loss(model, v)
            ll = lagrangian_loss(model, v)
            print("Loss: " * string(ol) * "     \t") 
            print("Lagrangian: " * string(ll) * "     \t") 
            println("Residual: " * string(res))
        end

        if res < model.tol
            break
        end

        v.μ = model.ρ * v.μ
    end

    L = size(v.Y, 2)
    Nₗ = size(v.T, 2)

    Xᵤ = v.X[:,Nₗ+1:end]
    Xᵤ_cut = [cut .- Nₗ for cut in v.X_cut[L+1:end]]
    raw_pred = bag_max(v.W'*Xᵤ .+ v.b, Xᵤ_cut)

    pred = CategoricalArray(onecold(raw_pred, levels(y)))

    return pred
end

mutable struct simmisvm_vars
    # Original vars
    Z::Array{Float64, 2}
    X::Array{Float64, 2}
    X_cut::Array{UnitRange{Int64}, 1}
    Y::Array{Float64, 2}
    W::Array{Float64, 2}
    b::Array{Float64, 1}
    S::Array{Float64, 2}
    𝓟_Ω::BitArray{2}

    # Introduced vars
    E::Array{Float64, 2}
    F::Array{Float64, 2}
    Q::Array{Float64, 2}
    R::Array{Float64, 2}
    T::Array{Float64, 2}
    U::Array{Float64, 2}

    # Lagrangian multiplers
    Λ::Array{Float64, 2}
    Π::Array{Float64, 2}
    Σ::Array{Float64, 2}
    Θ::Array{Float64, 2}
    Ω::Array{Float64, 2}
    Ξ::Array{Float64, 2}
    Δ::Array{Float64, 2}

    # Auxilary vars
    μ::Float64
    Xₗ::Array{Float64, 2}
    Xₗ_cut::Array{UnitRange{Int64}, 1}
    YI::Array{Float64, 2}
    WyX::Array{Float64, 2}
    by::Array{Float64, 2}
    rhs1::Array{Float64, 2}
    rhs2::Array{Float64, 2}
    rhs3::Array{Float64, 2}
end

function init_vars(model::SimMISVMClassifier, _X, _Xnew, _y)
    # Calculate auxilary variables
    P = length(_X) + length(_Xnew)
    L = length(_X)
    _Xall = vcat(_X, _Xnew)

    K = length(levels(_y))
    Nₗ = sum([nrows(x) for x in _X])
    np = [nrows(x) for x in _Xall]
    X_cut = [sum(np[1:n])-np[n]+1:sum(np[1:n]) for n in 1:length(np)]

    # Build Z by concatenating bags, build Y as onehot matrix
    Z = hcat([MLJBase.matrix(x)' for x in _Xall]...)
    Y = onehotbatch(_y, levels(_y)) .* 2.0 .- 1.0

    # Initialize some original vars
    d, N = size(Z)
    X = randn(d, N)
    S = randn(d, N)
    W = randn(d, K)
    b = randn(K)

    # Build missingness mask and replace NaN values in Z with 0
    𝓟_Ω = .!ismissing.(Z); Z[ismissing.(Z)] .= 0

    # Build fused lasso term. TODO: later.
    #dv = ones(n)
    #lv = vcat([vcat(-ones(size(x, 1)-1), 0) for x in _Xall]...)[1:n-1]
    #R = Bidiagonal(dv, lv, :L)

    # Introduced vars and associated lagrangian multipleirs
    E = randn(K, L); Λ = zeros(K, L)
    F = randn(d, N); Π = zeros(d, N) 
    Q = randn(K, L); Σ = zeros(K, L)
    R = randn(K, L); Ω = zeros(K, L)
    T = randn(K, Nₗ); Ξ = zeros(K, Nₗ)
    U = randn(K, Nₗ); Θ = zeros(K, Nₗ)

    # Other Lagrangian Multipliers
    Δ = zeros(d, N)
    
    # Auxilary variables
    Xₗ = randn(d, Nₗ)
    Xₗ_cut = X_cut[1:L]
    YI = hcat([repeat(Y[:,i], outer=(1, length(cut))) for (i, cut) in zip(1:N, Xₗ_cut)]...)
    WyX = randn(K, Nₗ)
    by = randn(K, Nₗ)

    rhs1 = zeros(size(X))
    rhs2 = zeros(size(Xₗ))
    rhs3 = zeros(size(Xₗ))

    v = simmisvm_vars(Z, X, X_cut, Y, W, b, S, 𝓟_Ω, E, F, Q, R, T, U, Λ, Π, Σ, Θ, Ω, Ξ, Δ, model.μ, Xₗ, Xₗ_cut, YI, WyX, by, rhs1, rhs2, rhs3)
    calc_Xₗ_WyX_and_by!(v)

    return v
end

function bag_max(WX, X_cut)
    return hcat([maximum(WX[:, cut], dims=2) for cut in X_cut]...)
end

function calc_Xₗ_WyX_and_by!(v::simmisvm_vars)
    K, Nₗ = size(v.T)
    v.Xₗ = v.X[:,1:Nₗ]
    WmX = v.W' * v.Xₗ
    bm = repeat(v.b, outer=(1, size(v.YI, 2)))

    v.WyX = repeat(WmX[v.YI .> 0]', outer=(K, 1))
    v.by = repeat(bm[v.YI .> 0]', outer=(K, 1))
end

function calc_residuals(model::SimMISVMClassifier, v::simmisvm_vars)
    calc_Xₗ_WyX_and_by!(v)

    E_res = v.E - (v.Y - v.Q + v.R)
    F_res = v.F - v.X
    Q_res = v.Q - bag_max(v.T, v.Xₗ_cut)
    T_res = v.T - (v.W' * v.Xₗ .+ v.b)
    R_res = v.R - bag_max(v.U, v.Xₗ_cut)
    U_res = v.U - (v.WyX + v.by)
    Z_res = v.Z - (v.X + v.S)

    return E_res, F_res, Q_res, T_res, R_res, U_res, Z_res
end

function obj_loss(model::SimMISVMClassifier, v::simmisvm_vars)
    l2reg = 0.5 * norm(v.W, 2)^2
    hinge = model.C * sum(max.(1 .- (bag_max(v.W'*v.Xₗ .+ v.b, v.Xₗ_cut) - bag_max(v.WyX + v.by, v.Xₗ_cut)).*v.Y, 0))
    trace = model.α * sum(svdvals(v.X))
    sparse = model.β * norm(v.𝓟_Ω .* v.S, 1)

    return l2reg + hinge + trace + sparse
end

function lagrangian_loss(model::SimMISVMClassifier, v::simmisvm_vars)
    l2reg = 0.5 * norm(v.W, 2)^2
    hinge = model.C * sum(max.(v.Y .* v.E, 0))
    trace = model.α * sum(svdvals(v.F))
    sparse = model.β * norm(v.𝓟_Ω .* v.S, 1)

    E_res, F_res, Q_res, T_res, R_res, U_res, Z_res = calc_residuals(model, v)

    Ediff = norm(E_res + v.Λ/v.μ, 2)^2
    Fdiff = norm(F_res + v.Π/v.μ, 2)^2
    Qdiff = norm(Q_res + v.Σ/v.μ, 2)^2
    Tdiff = norm(T_res + v.Θ/v.μ, 2)^2
    Rdiff = norm(R_res + v.Ω/v.μ, 2)^2
    Udiff = norm(U_res + v.Ξ/v.μ, 2)^2
    Zdiff = norm(Z_res + v.Δ/v.μ, 2)^2

    𝓛 = l2reg + hinge + trace + sparse + 0.5 * v.μ * (Ediff + Fdiff + Qdiff + Tdiff + Rdiff + Udiff + Zdiff)
end

function X_update!(model::SimMISVMClassifier, v::simmisvm_vars)
    K, Nₗ = size(v.Y)
    K, NI = size(v.YI)

    Wys = Array{Float64, 2}[]
    lhs = Array{Float64, 2}[]

    for m in 1:K
        Wy = repeat(v.W[:,m], outer=(1,K))
        push!(Wys, Wy)
        push!(lhs, inv(2*I + v.W * v.W' .+ Wy * Wy'))
    end

    v.rhs1 .= v.F .+ v.Π./v.μ .+ v.Z .- v.S .+ v.Δ./v.μ

    for (p, cut) in enumerate(v.X_cut)
        if p ≤ Nₗ
            p_prime = argmax(v.Y[:,p])
            by = v.b[p_prime]
            v.rhs2[:,cut] .= v.W * (v.T[:,cut] .- v.b .+ v.Θ[:,cut]./v.μ)
            v.rhs3[:,cut] .= Wys[p_prime] * (v.U[:,cut] .- by .+ v.Ξ[:,cut]./v.μ)
            v.X[:,cut] = lhs[p_prime] * (v.rhs1[:,cut] .+ v.rhs2[:,cut] .+ v.rhs3[:,cut])
        else
            v.X[:,cut] = 0.5*v.rhs1[:,cut]
        end
    end

    calc_Xₗ_WyX_and_by!(v)
end

function S_update!(model::SimMISVMClassifier, v::simmisvm_vars)
    M = v.Z - v.X + v.Δ/v.μ
    Mmid = -model.β/v.μ .<= M .<= model.β/v.μ
    Mgt = M .> model.β/v.μ
    Mlt = M .< -model.β/v.μ

    v.S = v.𝓟_Ω .* (M .* .!Mmid - Mgt * (model.β/v.μ) + Mlt * (model.β/v.μ)) + (.!v.𝓟_Ω) .* M
end

function W_update!(model::SimMISVMClassifier, v::simmisvm_vars)
    K, NI = size(v.YI)

    lhs1 = sum([(v.T[:,cut] .- v.b + v.Θ[:,cut]/v.μ) * v.X[:,cut]' for cut in v.Xₗ_cut])
    rhs1 = sum([v.X[:,cut]*v.X[:,cut]' for cut in v.Xₗ_cut])

    for m in 1:K
        lhs2 = sum([((v.U[:,cut] - v.by[:,cut] + v.Ξ[:,cut]/v.μ) * v.X[:,cut]') * (v.Y[:,n][m] > 0) for (n, cut) in enumerate(v.Xₗ_cut)])

        step1 = [v.X[:,cut][:,v.YI[:,cut][m,:] .> 0] for cut in v.Xₗ_cut]
        rhs2 = sum([x * x' for x in step1])

        v.W[:,m] = (I/v.μ + rhs1 + K*rhs2)' \ (lhs1[m,:]' + sum(lhs2, dims=1))'
    end
    calc_Xₗ_WyX_and_by!(v)
end

function b_update!(model::SimMISVMClassifier, v::simmisvm_vars)
    K, NI = size(v.YI)

    numer1 = sum(v.T - v.W'*v.Xₗ + v.Θ/v.μ, dims=2)
    numer2 = zeros(size(numer1))
    for m in 1:K
        prime = v.YI[m,:] .> 0
        numer2[m] = sum((v.U - v.WyX + v.Ξ/v.μ)[:,prime])
    end
    numer = numer1 + numer2
    denom = float(NI .+ K * sum(v.YI .> 0.0, dims=2))

    v.b = vec(numer ./ denom)
    calc_Xₗ_WyX_and_by!(v)
end


function E_update!(model::SimMISVMClassifier, v::simmisvm_vars)
    N = v.Y - v.Q + v.R - v.Λ/v.μ
    gt = (v.Y .* N) .> model.C/v.μ
    mid = 0 .<= (v.Y .* N) .<= model.C/v.μ

    v.E = N .* .!mid - gt .* v.Y .* (model.C/v.μ)
end

"""
A singular value thresholding algorithm to solve

min_X α‖X‖_* + μ/2‖X - A‖_F^2

where δ = α/μ

@article{cai2010singular,
  title={A singular value thresholding algorithm for matrix completion},
  author={Cai, Jian-Feng and Cand{\`e}s, Emmanuel J and Shen, Zuowei},
  journal={SIAM Journal on optimization},
  volume={20},
  number={4},
  pages={1956--1982},
  year={2010},
  publisher={SIAM}
}
"""
function svt(A, δ)
    u, s, v = svd(A)
    return u * diagm(0 => max.(s .- δ, 0)) * v'
end

function F_update!(model::SimMISVMClassifier, v::simmisvm_vars)
    O = v.X - v.Π/v.μ
    v.F = svt(O, model.α/v.μ)
end

function Q_update!(model::SimMISVMClassifier, v::simmisvm_vars)
    v.Q = 0.5 * (v.Y - v.E + v.R - v.Λ/v.μ + bag_max(v.T, v.Xₗ_cut) - v.Σ/v.μ)
end

function R_update!(model::SimMISVMClassifier, v::simmisvm_vars)
    v.R = 0.5 * (v.E - v.Y + v.Q + v.Λ/v.μ + bag_max(v.U, v.Xₗ_cut) - v.Ω/v.μ)
end

function T_update!(model::SimMISVMClassifier, v::simmisvm_vars)
    K = size(v.Y, 1)
    Φ = v.W' * v.Xₗ .+ v.b - v.Θ/v.μ
    for (i, cut) in enumerate(v.Xₗ_cut)
        ni = length(cut)
        for m in 1:K
            ϕᵢₘ = Φ[m, cut]
            v.T[m,cut] = ϕᵢₘ
            v.T[m,cut[1]+argmax(ϕᵢₘ)-1] = 0.5 * (maximum(ϕᵢₘ) + v.Q[m, i] + v.Σ[m, i]/v.μ)
        end
    end
end

function U_update!(model::SimMISVMClassifier, v::simmisvm_vars)
    K = size(v.Y, 1)
    Ψ = v.WyX + v.by - v.Ξ/v.μ
    for (i, cut) in enumerate(v.Xₗ_cut)
        ni = length(cut)
        for m in 1:K
            ψᵢₘ = Ψ[m, cut]
            v.U[m,cut] = ψᵢₘ
            v.U[m,cut[1]+argmax(ψᵢₘ)-1] = 0.5 * (maximum(ψᵢₘ) + v.R[m, i] + v.Ω[m, i]/v.μ)
        end
    end
end
