using DrWatson
using MLJ
using MLJBase
using Impute: LOCF, Fill
using Random

Random.seed!(1234)

include(projectdir("data_utils.jl"))
include(srcdir("SimMISVM.jl"))

X, y = @load_covid()

function precis(ŷ, y) 
    return tpr(ŷ, y) / (tpr(ŷ, y) + fpr(ŷ, y))
end

measure = [precis, tpr, f1score, accuracy]
rcv = CV(nfolds=6, shuffle=true)

evals_and_models = []

imp = LOCF() ∘ Fill(value=0.0)
hours = 24.0

# ------ KNN ------
KNNClassifier = @load KNNClassifier
knn = KNNClassifier(K=7)
knnp = @pipeline knn yhat->mode.(yhat)
ni_knn = COVIDInstanceClassifier(hours=hours, mi2si=true, impute_chain=imp, classifier=knnp)
knn_machine = machine(ni_knn, X, y)
knn_eval = evaluate!(knn_machine, resampling=rcv, measure=measure)
push!(evals_and_models, (knn_eval, "kNN"))
@show knn_eval

# ------ Gradient Boosted Trees ------
XGBoostClassifier = @load XGBoostClassifier
xgboost = XGBoostClassifier(eta=0.22, max_depth=8, lambda=32, alpha=0.44)
xgboostp = @pipeline xgboost yhat->mode.(yhat)
ni_xgboost = COVIDInstanceClassifier(hours=hours, mi2si=true, impute_chain=imp, classifier=xgboostp)
xgboost_machine = machine(ni_xgboost, X, y)
xgboost_eval = evaluate!(xgboost_machine, resampling=rcv, measure=measure)
push!(evals_and_models, (xgboost_eval, "XGBoost"))
@show xgboost_eval

# ------ LightGBM ------
LGBMClassifier = @load LGBMClassifier
lgbm = LGBMClassifier(learning_rate=0.27, num_leaves=32, max_depth=24)
lgbmp = @pipeline lgbm yhat->mode.(yhat)
ni_lgbm = COVIDInstanceClassifier(hours=hours, mi2si=true, impute_chain=imp, classifier=lgbmp)
lgbm_machine = machine(ni_lgbm, X, y)
lgbm_eval = evaluate!(lgbm_machine, resampling=rcv, measure=measure)
push!(evals_and_models, (lgbm_eval, "LightGBM"))
@show lgbm_eval

# ------ SVM ------
SVMClassifier = @load SVMClassifier
svm = SVMClassifier(C=5e4, kernel="linear")
ni_svm = COVIDInstanceClassifier(hours=hours, mi2si=true, impute_chain=imp, classifier=svm)
svm_machine = machine(ni_svm, X, y)
svm_eval = evaluate!(svm_machine, resampling=rcv, measure=measure)
push!(evals_and_models, (svm_eval, "SVM"))
@show svm_eval

# ------ Ours (SimMISVM) ------
simmisvm = SimMISVMClassifier(C=10, α=0.01, β=0.01, μ=1e-4, ρ=1.2, maxiter=1000, tol=1e-4)
ni_simmisvm = COVIDInstanceClassifier(hours=hours, mi2si=false, impute_chain=missing, classifier=simmisvm)
simmisvm_machine = machine(ni_simmisvm, X, y)
simmisvm_eval = evaluate!(simmisvm_machine, resampling=rcv, measure=measure, verbosity=1)
@show simmisvm_eval
push!(evals_and_models, (simmisvm_eval, "SimMISVM"))

function mean_and_std(per_fold)
    return string(round(mean(per_fold), digits=3)) * "," * string(round(std(per_fold), digits=3))
end

println("model,precision,pstd,recall,rstd,f1score,fstd,accuracy,astd,hour")
for tuple in evals_and_models
    i, j = tuple
    print(j, ",")
    print(mean_and_std(i.per_fold[1]))
    print(",")
    print(mean_and_std(i.per_fold[2]))
    print(",")
    print(mean_and_std(i.per_fold[3]))
    print(",")
    print(mean_and_std(i.per_fold[4]))
    print(",")
    println(hours)
end
