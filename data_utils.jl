using DrWatson
using DataFrames
using CSV
using MLJ: OpenML, Standardizer
using MLJBase: ordered!
using Dates
using Impute: Chain, impute

macro load_covid()
    """ Loads the COVID-19 Timeseries Dataset

    Source: https://www.nature.com/articles/s42256-020-0180-7
    """

    N_PATIENTS = 375
    df = CSV.File(datadir("time_series_375_prerpocess_en.csv")) |> DataFrame

    # X in matrix form
    X = DataFrames.select(df, :PATIENT_ID, Symbol("Admission time"), :RE_DATE, :age, :gender, :outcome, 8:size(df, 2))
    DataFrames.select!(X, Not("2019-nCoV nucleic acid detection"))

    # Remove rows with no RE_DATE, then remove outcome from X after saving in y
    delete!(X, ismissing.(X.RE_DATE))
    y = X.outcome
    DataFrames.select!(X, Not("outcome"))

    # Convert string dates to DateTime
    format = DateFormat("Y-m-d H:M:S")
    X."Admission time" = DateTime.(X."Admission time", format)
    X.RE_DATE = DateTime.(X.RE_DATE, format)

    # Identify the start and end indices for each patient (timeseries)
    X_cut = []
    start = 1
    for i in 2:length(X.PATIENT_ID)
        if !ismissing(X.PATIENT_ID[i])
            append!(X_cut, [start:i-1])
            start=i
        end
    end
    append!(X_cut, [start:length(X.PATIENT_ID)])
    DataFrames.select!(X, Not("PATIENT_ID"))

    # Get range of dynamic biomarkers (skip Admission time, RE_DATE, age, and gender)
    dynamic_biomarkers = 5:size(X, 2)

    # Replace missing with mean and then Standardize X. Once done,
    # replace missing with NaN values.
    X_missing = ismissing.(X)

    for n in names(X)[dynamic_biomarkers]
        m = mean(skipmissing(X[!, n]))
        X[!, n] = passmissing(convert).(Float64, X[!, n])
        replace!(X[!, n], missing=>m)
        X[!, n] = convert(Array{Float64}, X[!, n])
    end

    # Normalze data
    stand_model = Standardizer()
    X = MLJBase.transform(fit!(machine(stand_model, X)), X)

    # Replace missing with NaN (of type Float64)
    for n in names(X)[dynamic_biomarkers]
        X[!, n] = convert(Array{Union{Float64, Missing}}, X[!, n])
        X[!, n][X_missing[!, n]] .= missing
    end

    # y in vector form, X in bag form
    y_final = vcat([levels(y[cut,:]) for cut in X_cut]...)
    X_final = [X[cut,:] for cut in X_cut]
    X_missing = [X_missing[cut,:] for cut in X_cut]

    @assert length(X_final) == length(y_final)

    return X_final, ordered!(CategoricalArray(y_final), true)
end

mutable struct COVIDInstanceClassifier <: Deterministic
    hours::Float64
    mi2si::Bool
    impute_chain::Union{Chain, Missing}
    classifier
end

function COVIDInstanceClassifier(; hours=24.0, mi2si=true, impute_chain=missing, classifier=missing)
    model = COVIDInstanceClassifier(hours, mi2si, impute_chain, classifier)
end

function MLJBase.fit(model::COVIDInstanceClassifier, verbosity::Integer, X, y)
    X = [copy(x) for x in X]
    remove_static!(model, X)

    if model.mi2si
        Xsi = multi_instance_to_single_instance(model, X)
        return MLJBase.fit(model.classifier, verbosity, Xsi, y)
    elseif !ismissing(model.impute_chain)
        Xim = impute_multi_instance(model, X)
        return MLJBase.fit(model.classifier, verbosity, Xim, y)
    else
        return MLJBase.fit(model.classifier, verbosity, X, y)
    end
end

function MLJBase.predict(model::COVIDInstanceClassifier, fitresult, Xnew)
    Xc = cut_by_time_after_admission(model, Xnew)
    remove_static!(model, Xc)

    if model.mi2si
        Xsi = multi_instance_to_single_instance(model, Xc)
        return MLJBase.predict(model.classifier, fitresult, Xsi)
    elseif !ismissing(model.impute_chain)
        Xim = impute_multi_instance(model, Xc)
        return MLJBase.predict(model.classifier, fitresult, Xim)
    else
        return MLJBase.predict(model.classifier, fitresult, Xc)
    end
end

function multi_instance_to_single_instance(model::COVIDInstanceClassifier, X)
    X_single_instance = DataFrame(zeros(0, size(X[1], 2)), names(X[1]))
    for x in X
        x = Matrix(x)
        row1 = @view x[1,:]
        row1[ismissing.(row1)] .= 0.0
        x_imp = impute(x, model.impute_chain)
        push!(X_single_instance, x_imp[end,:])
    end
    return X_single_instance
end

function impute_multi_instance(model::COVIDInstanceClassifier, X)
    X_impute_missing = DataFrame[]
    for x in X
        x = Matrix(x)
        row1 = @view x[1,:]
        row1[ismissing.(row1)] .= 0.0
        x_imp = impute(x, model.impute_chain)
        push!(X_impute_missing, DataFrame(x_imp, names(X[1])))
    end
    return X_impute_missing
end

function cut_by_time_after_admission(model::COVIDInstanceClassifier, Xs)
    MILLISECONDS_IN_AN_HOUR = 3.6e6
    Xs_cut2day = DataFrame[]
    for X in Xs
        num2keep = 1
        for row in eachrow(X) 
            t1 = row."Admission time"
            t2 = row.RE_DATE
            if (Dates.value(t2-t1) / MILLISECONDS_IN_AN_HOUR) ≤ model.hours
                num2keep = num2keep+1
            end
        end
        if num2keep > nrows(X)
            num2keep = num2keep-1
        end
        push!(Xs_cut2day, DataFrame(X[1:num2keep, :]))
    end
    return Xs_cut2day
end

function remove_static!(model::COVIDInstanceClassifier, X)
    for x in X
        DataFrames.select!(x, Not(["Admission time"]))
        DataFrames.select!(x, Not([:RE_DATE, :age, :gender]))
    end
end


