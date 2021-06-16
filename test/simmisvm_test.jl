using DrWatson
using Test
using MLJ

include(srcdir("SimMISVM.jl"))

function init_dummy_data()
    bag1_3instance = [1.0  1.0  100.0;
                      1.0  2.0  0.0;
                      0.0  0.0  0.0]
    bag2_2instance = [1.0  2.0  100.0;
                      2.0  2.0 -2.0]
    bag3_1instance = [3.0  3.0 -100.0]
    bag4_1instance = [4.0  4.0 -100.0]

    X = [bag1_3instance, bag2_2instance, bag3_1instance, bag4_1instance]
    y = vec([0 0 1 1])

    unlabled_bag1_2instance = [1.0     missing   100.0;
                               missing missing missing]
    unlabeled_bag2_1instance = [missing missing -100.0;]

    Xnew = [unlabled_bag1_2instance, unlabeled_bag2_1instance]

    return X, Xnew, y
end

function init_dummy_model_and_var()
    model = SimMISVMClassifier(C=3.0, α=2.0, β=4.0)
    X, Xnew, y = init_dummy_data()

    v = init_vars(model, X, Xnew, y)

    v.Λ .= 1.0
    v.Π .= 2.0
    v.Σ .= 3.0
    v.Θ .= 4.0
    v.Ω .= 5.0
    v.Ξ .= 6.0
    v.Δ .= 7.0

    v.μ = 2.0
	
    return model, v
end

@testset "check init" begin
    model, v = init_dummy_model_and_var()

    @test size(v.Z) == (3, 10)
    @test size(v.X) == (3, 10)
    @test v.X_cut == [1:3, 4:5, 6:6, 7:7, 8:9, 10:10]
    @test v.Y == [ 1.0  1.0 -1.0 -1.0;
                  -1.0 -1.0  1.0  1.0]
    @test size(v.W) == (3, 2)
    @test size(v.b) == (2,)
    @test size(v.S) == (3, 10)
    @test size(v.𝓟_Ω) == (3, 10)
    
    @test size(v.E) == (2, 4)
    @test size(v.F) == (3, 10)
    @test size(v.Q) == (2, 4)
    @test size(v.R) == (2, 4)
    @test size(v.T) == (2, 7)
    @test size(v.U) == (2, 7)

    @test size(v.Λ) == (2, 4)
    @test size(v.Π) == (3, 10)
    @test size(v.Σ) == (2, 4)
    @test size(v.Ω) == (2, 4)
    @test size(v.Θ) == (2, 7)
    @test size(v.Ξ) == (2, 7)
end

@testset "check losses" begin
    @testset "check obj loss" begin
        model, v = init_dummy_model_and_var()
        @test obj_loss(model, v) > 0.0
    end

    @testset "check lagrangian loss" begin
        model, v = init_dummy_model_and_var()
        @test lagrangian_loss(model, v) > 0.0
    end
end

function get_plus_minus_evals(var_to_check, model::SimMISVMClassifier, v::simmisvm_vars)
    δ = .0001

    lower_bound = lagrangian_loss(model, v)
    var_to_check .+= δ
    plus_cost = lagrangian_loss(model, v)
    var_to_check .-= 2.0*δ
    minus_cost = lagrangian_loss(model, v)

    return lower_bound, minus_cost, plus_cost
end

@testset "check updates" begin
    @testset "check X update" begin
        model, v = init_dummy_model_and_var()
        l1 = lagrangian_loss(model, v)
        X_update!(model, v)
        l2 = lagrangian_loss(model, v)
        @test l2 < l1
        lower, minus, plus = get_plus_minus_evals(v.X, model, v)
        @test lower < minus
        @test lower < plus
    end

    @testset "check S update" begin
        model, v = init_dummy_model_and_var()
        l1 = lagrangian_loss(model, v)
        S_update!(model, v)
        l2 = lagrangian_loss(model, v)
        @test l2 < l1
        lower, minus, plus = get_plus_minus_evals(v.S, model, v)
        @test lower < minus
        @test lower < plus
    end

    @testset "check W update" begin
        model, v = init_dummy_model_and_var()
        l1 = lagrangian_loss(model, v)
        W_update!(model, v)
        l2 = lagrangian_loss(model, v)
        @test l2 < l1
        lower, minus, plus = get_plus_minus_evals(v.W, model, v)
        @test lower < minus
        @test lower < plus
    end

    @testset "check b update" begin
        model, v = init_dummy_model_and_var()
        l1 = lagrangian_loss(model, v)
        b_update!(model, v)
        l2 = lagrangian_loss(model, v)
        @test l2 < l1
        lower, minus, plus = get_plus_minus_evals(v.b, model, v)
        @test lower < minus
        @test lower < plus
    end

    @testset "check E update" begin
        model, v = init_dummy_model_and_var()
        l1 = lagrangian_loss(model, v)
        E_update!(model, v)
        l2 = lagrangian_loss(model, v)
        @test l2 < l1
        lower, minus, plus = get_plus_minus_evals(v.E, model, v)
        @test lower < minus
        @test lower < plus
    end

    @testset "check F update" begin
        model, v = init_dummy_model_and_var()
        l1 = lagrangian_loss(model, v)
        F_update!(model, v)
        l2 = lagrangian_loss(model, v)
        @test l2 < l1
        lower, minus, plus = get_plus_minus_evals(v.F, model, v)
        @test lower < minus
        @test lower < plus
    end

    @testset "check Q update" begin
        model, v = init_dummy_model_and_var()
        l1 = lagrangian_loss(model, v)
        Q_update!(model, v)
        l2 = lagrangian_loss(model, v)
        @test l2 < l1
        lower, minus, plus = get_plus_minus_evals(v.Q, model, v)
        @test lower < minus
        @test lower < plus
    end

    @testset "check R update" begin
        model, v = init_dummy_model_and_var()
        l1 = lagrangian_loss(model, v)
        R_update!(model, v)
        l2 = lagrangian_loss(model, v)
        @test l2 < l1
        lower, minus, plus = get_plus_minus_evals(v.R, model, v)
        @test lower < minus
        @test lower < plus
    end

    @testset "check T update" begin
        model, v = init_dummy_model_and_var()
        l1 = lagrangian_loss(model, v)
        T_update!(model, v)
        l2 = lagrangian_loss(model, v)
        @test l2 < l1
        lower, minus, plus = get_plus_minus_evals(v.T, model, v)
        @test lower < minus
        @test lower < plus
    end

    @testset "check U update" begin
        model, v = init_dummy_model_and_var()
        l1 = lagrangian_loss(model, v)
        U_update!(model, v)
        l2 = lagrangian_loss(model, v)
        @test l2 < l1
        lower, minus, plus = get_plus_minus_evals(v.U, model, v)
        @test lower < minus
        @test lower < plus
    end
end

@testset "check ml functions" begin
    @testset "check fit and predict" begin
        model = SimMISVMClassifier()
        X, Xnew, y = init_dummy_data()
        misvm = machine(model, X, y)

        fit!(misvm, verbosity=1)
        pred = predict(misvm, Xnew)

        @test pred == vec([0 1])
    end
end
